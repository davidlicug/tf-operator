// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow

import (
	"sync"
	"strings"

	"k8s.io/client-go/tools/cache"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	corelisterv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/informers"
	"errors"
	tfv1 "github.com/kubeflow/tf-operator/pkg/apis/tensorflow/v1"
	"sort"
)

// A set of port allocations for a node
type portAllocation map[int32]bool

type pn struct {
	pa   portAllocation
	port int32
}

// PortAllocator manages the dynamic port
// allocation strategy. Only use exposed methods to ensure
// appropriate locking is taken.
// The PortAllocator does not currently support mixing static portAllocations (or any pods with defined HostPort)
// within the dynamic port range other than the ones it coordinates.
type PortAllocator struct {
	mutex              	sync.RWMutex
	portAllocations    	[]portAllocation
	minPort            	int32
	maxPort            	int32
	podInformerSynced   cache.InformerSynced
	podLister   		corelisterv1.PodLister
	podInformer			cache.SharedIndexInformer
	nodeInformerSynced  cache.InformerSynced
	nodeLister         	corelisterv1.NodeLister
	nodeInformer       	cache.SharedIndexInformer
}

// NewPortAllocator returns a new dynamic port
// allocator. minPort and maxPort are the top and bottom portAllocations that can be allocated in the range for
// the game servers
func NewPortAllocator(minPort, maxPort int32, kubeInformerFactory informers.SharedInformerFactory) *PortAllocator {

	v1 := kubeInformerFactory.Core().V1()
	nodes := v1.Nodes()
	pods := v1.Pods()

	pa := &PortAllocator{
		mutex:              sync.RWMutex{},
		minPort:            minPort,
		maxPort:            maxPort,
		podInformerSynced:  pods.Informer().HasSynced,
		podLister:   		pods.Lister(),
		podInformer: 		pods.Informer(),
		nodeLister:         nodes.Lister(),
		nodeInformer:       nodes.Informer(),
		nodeInformerSynced: nodes.Informer().HasSynced,
	}

	pa.podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		DeleteFunc: pa.syncDeletePodPort,
	})

	return pa
}

// Run sets up the current state of port allocations and
// starts tracking Pod and Node changes
func (pa *PortAllocator) Run(stop <-chan struct{}) error {
	if !cache.WaitForCacheSync(stop, pa.podInformerSynced, pa.nodeInformerSynced) {
		return errors.New("failed to wait for caches to sync")
	}

	// on run, let's make sure we start with a perfect slate straight away
	if err := pa.syncAll(); err != nil {
		return err
	}

	return nil
}

// syncAll syncs the pod, node and gameserver caches then
// traverses all Nodes in the cluster and all looks at GameServers
// and Terminating Pods values make sure those
// portAllocations are marked as taken.
// Locks the mutex while doing this.
// This is basically a stop the world Garbage Collection on port allocations, but it only happens on startup.
func (pa *PortAllocator) syncAll() error {
	pa.mutex.Lock()
	defer pa.mutex.Unlock()

	nodes, err := pa.nodeLister.List(labels.Everything())
	if err != nil {
		return errors.New("error listing all nodes")
	}

	pods, err := pa.podLister.List(labels.Everything())
	if err != nil {
		return errors.New("error listing all GameServers")
	}

	// place to put pod port allocations that are not ready yet/after the ready state
	allocations, nonReadyNodesPorts := pa.registerExistingPodPorts(pods, nodes)

	// close off the port on the first node you find
	// we actually don't mind what node it is, since we only care
	// that there is a port open *somewhere* as the default scheduler
	// will re-route for us based on HostPort allocation
	for _, p := range nonReadyNodesPorts {
		allocations = setPortAllocation(p, allocations, true)
	}

	pa.portAllocations = allocations

	return nil
}

// registerExistingPodPorts registers the pod ports against nodePorts.
// and returns an ordered list of portAllocations per cluster nodes, and an array of
// any pod allocated a port, but not yet assigned a Node will returned as an array of port values.
func (pa *PortAllocator) registerExistingPodPorts(pods []*corev1.Pod, nodes []*corev1.Node) ([]portAllocation, []int32) {
	// setup blank port values
	nodePortAllocation := pa.nodePortAllocation(nodes)
	nodePortCount := make(map[string]int64, len(nodes))
	for _, n := range nodes {
		nodePortCount[n.ObjectMeta.Name] = 0
	}

	var nonReadyNodesPorts []int32

	for _, pod := range pods {
		if !pod.Spec.HostNetwork {
			continue
		}
		for _, container := range pod.Spec.Containers {
			for _, p := range container.Ports {
				// if the node doesn't exist, it's likely unscheduled
				_, ok := nodePortAllocation[pod.Spec.NodeName]
				if pod.Spec.NodeName != "" && ok {
					nodePortAllocation[pod.Spec.NodeName][p.HostPort] = true
					nodePortCount[pod.Spec.NodeName]++
				} else if p.HostPort != 0 {
					nonReadyNodesPorts = append(nonReadyNodesPorts, p.HostPort)
				}
			}
		}
	}

	// make a list of the keys
	keys := make([]string, 0, len(nodePortAllocation))
	for k := range nodePortAllocation {
		keys = append(keys, k)
	}

	// sort, since this is how it would have originally been allocated across the
	// ordered []portAllocation
	sort.Slice(keys, func(i, j int) bool {
		return nodePortCount[keys[i]] > nodePortCount[keys[j]]
	})

	// this gives us back an ordered node list
	allocations := make([]portAllocation, len(nodePortAllocation))
	for i, k := range keys {
		allocations[i] = nodePortAllocation[k]

	}

	return allocations, nonReadyNodesPorts
}

// Allocate assigns a port to the tfJob and returns it.
// Return ErrPortNotFound if no port is allocatable
func (pa *PortAllocator) Allocate(tfJob *tfv1.TFJob) map[string][]pn {
	pa.mutex.Lock()
	defer pa.mutex.Unlock()
	// we only want this to be called inside the mutex lock
	// so let's define the function here so it can never be called elsewhere.
	// Also the return gives an escape from the double loop
	findOpenPorts := func(amount int) []pn {
		var ports []pn
		for _, n := range pa.portAllocations {
			for p, taken := range n {
				if !taken {
					ports = append(ports, pn{pa: n, port: p})
					// only allocate as many ports as are asked for by the tfJob
					if len(ports) == amount {
						return ports
					}
				}
			}
		}
		return ports
	}

	// this allows us to do recursion, within the mutex lock
	var allocate func(amount int) []pn
	allocate = func(amount int) []pn {
		allocations := findOpenPorts(amount)

		if len(allocations) == amount {
			return allocations
		}

		// if we get here, we ran out of ports. Add a node, and try again.
		// this is important, because to autoscale scale up, we create tfJob that
		// can't be scheduled on the current set of nodes, so we need to be sure
		// there are always ports available to be allocated.
		pa.portAllocations = append(pa.portAllocations, pa.newPortAllocation())

		return allocate(amount)
	}

	mport := make(map[string][]pn)
	for role, spec := range tfJob.Spec.TFReplicaSpecs {
		if spec.Template.Spec.HostNetwork {
			found := false
			for _, container := range spec.Template.Spec.Containers {
				for _, port := range container.Ports{
					if port.Name == tfv1.DefaultPortName && port.ContainerPort == tfv1.DefaultPort{
						found = true
						break
					}
				}
				if found {
					break
				}
			}
			if found {
				rtype := strings.ToLower(string(role))
				mport[rtype] = allocate(int(*spec.Replicas))
			}
		}
	}


	return mport
}

// DeAllocate marks the given port as no longer allocated
func (pa *PortAllocator) UndoAllocate(pns []pn) {
	// skip if it wasn't previously allocated

	pa.mutex.Lock()
	defer pa.mutex.Unlock()

	for _, pn := range pns {
		if pn.port < pa.minPort || pn.port > pa.maxPort {
			continue
		}
		pa.portAllocations = setPortAllocation(pn.port, pa.portAllocations, false)
	}
}

func (pa *PortAllocator) syncDeletePodPort(object interface{}) {
	if pod, ok := object.(*corev1.Pod); ok {
		pa.DeAllocate(pod)
	}
}

// DeAllocate marks the given port as no longer allocated
func (pa *PortAllocator) DeAllocate(pod *corev1.Pod) {
	// skip if it wasn't previously allocated

	pa.mutex.Lock()
	defer pa.mutex.Unlock()

	if pod.Spec.HostNetwork {
		for _, container := range pod.Spec.Containers {
			for _, p := range container.Ports {
				if p.HostPort < pa.minPort || p.HostPort > pa.maxPort {
					continue
				}
				pa.portAllocations = setPortAllocation(p.HostPort, pa.portAllocations, false)
			}
		}
	}
}



// nodePortAllocation returns a map of port allocations all set to being available
// with a map key for each node, as well as the node registry record (since we're already looping)
func (pa *PortAllocator) nodePortAllocation(nodes []*corev1.Node) map[string]portAllocation {
	nodePorts := map[string]portAllocation{}

	for _, n := range nodes {
		// ignore unschedulable nodes
		if !n.Spec.Unschedulable {
			nodePorts[n.Name] = pa.newPortAllocation()
		}
	}

	return nodePorts
}

func (pa *PortAllocator) newPortAllocation() portAllocation {
	p := make(portAllocation, (pa.maxPort-pa.minPort)+1)
	for i := pa.minPort; i <= pa.maxPort; i++ {
		p[i] = false
	}

	return p
}

// setPortAllocation takes a port from an all
func setPortAllocation(port int32, allocations []portAllocation, taken bool) []portAllocation {
	for _, np := range allocations {
		if np[port] != taken {
			np[port] = taken
			break
		}
	}
	return allocations
}
