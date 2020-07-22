// Copyright 2018 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package controller provides a Kubernetes controller for a TFJob resource.
package tensorflow

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	tfv1alpha2 "github.com/kubeflow/tf-operator/pkg/apis/tensorflow/v1alpha2"
	"github.com/kubeflow/tf-operator/pkg/controller.v2/jobcontroller"
)

// TFConfig is a struct representing the distributed TensorFlow config.
// This struct is turned into an environment variable TF_CONFIG
// which is used by TensorFlow processes to configure themselves.
// https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig#methods
// https://cloud.google.com/ml-engine/docs/tensorflow/distributed-training-details
type TFConfig struct {
	// Cluster represents a TensorFlow ClusterSpec.
	// See: https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec
	Cluster ClusterSpec `json:"cluster"`
	Task    TaskSpec    `json:"task"`
	// Environment is used by tensorflow.contrib.learn.python.learn in versions <= 1.3
	// TODO(jlewi): I don't think it is used in versions TF >- 1.4. So we can eventually get rid of it.
	Environment string `json:"environment"`
}

// ClusterSpec represents a cluster TensorFlow specification.
// https://www.tensorflow.org/deploy/distributed#create_a_tftrainclusterspec_to_describe_the_cluster
// It is a map from job names to network addresses.
type ClusterSpec map[string][]string

// TaskSpec is the specification for a task (PS or worker) of the TFJob.
type TaskSpec struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

// genTFConfig will generate the environment variable TF_CONFIG
// {
//     "cluster": {
//         "ps": ["ps1:2222", "ps2:2222"],
//         "worker": ["worker1:2222", "worker2:2222", "worker3:2222"]
//     },
//     "task": {
//         "type": "ps",
//         "index": 1
//         },
//     }
// }
func genTFConfigJSONStr(tfjob *tfv1alpha2.TFJob, rtype, index string) (string, error) {
	// Configure the TFCONFIG environment variable.
	i, err := strconv.ParseInt(index, 0, 32)
	if err != nil {
		return "", err
	}

	cluster, err := genClusterSpec(tfjob)
	if err != nil {
		return "", err
	}

	tfConfig := TFConfig{
		Cluster: cluster,
		Task: TaskSpec{
			Type:  rtype,
			Index: int(i),
		},
		// We need to set environment to cloud  otherwise it will default to local which isn't what we want.
		// Environment is used by tensorflow.contrib.learn.python.learn in versions <= 1.3
		// TODO(jlewi): I don't think it is used in versions TF >- 1.4. So we can eventually get rid of it.
		Environment: "cloud",
	}

	tfConfigJSONStr, err := json.Marshal(tfConfig)
	if err != nil {
		return "", err
	}

	return string(tfConfigJSONStr), nil
}

// genClusterSpec will generate ClusterSpec.
func genClusterSpec(tfjob *tfv1alpha2.TFJob) (ClusterSpec, error) {
	clusterSpec := make(ClusterSpec)

	for rtype, spec := range tfjob.Spec.TFReplicaSpecs {
		if rtype == tfv1alpha2.TFReplicaTypeEval {
			// https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
			// evaluator is not part of training cluster
			continue
		}
		rt := strings.ToLower(string(rtype))
		replicaNames := make([]string, 0, *spec.Replicas)

		port, err := GetPortFromTFJob(tfjob, rtype)
		if err != nil {
			return nil, err
		}
		if spec.Template.Spec.HostNetwork && port == tfv1alpha2.DefaultPort {
			for i := int32(0); i < *spec.Replicas; i++ {
				if portStr, ok := tfjob.Annotations[rt]; ok {
					ports := strings.Split(portStr, ",")
					if i < int32(len(ports)){
						value, _ := strconv.Atoi(ports[i])
						if value != 0 {
							port = int32(value)
						}
					}
				}
				host := fmt.Sprintf("%s:%d", jobcontroller.GenGeneralName(tfjob.Name, rt, fmt.Sprintf("%d", i)), port)
				replicaNames = append(replicaNames, host)
			}
			clusterSpec[rt] = replicaNames
		}else {
			for i := int32(0); i < *spec.Replicas; i++ {
				host := fmt.Sprintf("%s:%d", jobcontroller.GenGeneralName(tfjob.Name, rt, fmt.Sprintf("%d", i)), port)
				replicaNames = append(replicaNames, host)
			}
			clusterSpec[rt] = replicaNames
		}

	}

	return clusterSpec, nil
}
