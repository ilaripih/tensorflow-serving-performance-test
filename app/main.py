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
    # The "raw_detection_boxes" and "raw_detection_scores" tensors
    # bloat the response size.
    channel = grpc.insecure_channel('local_serving:8500', options=[
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
    ])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    batch_size = 4
    n_requests = 25
    start_time = time.time()
    for _ in range(n_requests):
        predict(stub, np.zeros(shape=(batch_size, 640, 640, 3), dtype=np.uint8))
    elapsed_time = time.time() - start_time
    n_images = batch_size * n_requests

    # Results with GTX 1060 (6GB):
    # Serving 1.15.0-gpu: 0.22 FPS
    # Serving 1.14.0-gpu: 4.19 FPS
    # Serving 2.0.0-gpu: 4.05 FPS
    print('{} images in {:.2f} ({:.2f} FPS)'.format(
        n_images, elapsed_time, n_images / elapsed_time))


if __name__ == '__main__':
    tf.compat.v1.app.run()
