#!/usr/bin/env python3.6

from __future__ import print_function

import grpc
import time
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def predict(stub, np_images):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'retinanet'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(np_images, shape=np_images.shape))
    stub.Predict(request, 30.0) # 30 seconds


def main(_):
    channel = grpc.insecure_channel('local_serving:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    batch_size = 4
    n_requests = 250
    start_time = time.time()
    for _ in range(n_requests):
        predict(stub, np.zeros(shape=(batch_size, 640, 640, 3), dtype=np.uint8))
    elapsed_time = time.time() - start_time
    n_images = batch_size * n_requests
    print('{} images in {:.2f} ({:.2f} FPS)'.format(
        n_images, elapsed_time, n_images / elapsed_time))


if __name__ == '__main__':
    tf.compat.v1.app.run()
