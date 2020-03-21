import numpy as np
import cv2
import tensorflow as tf

#single image capture
cap = cv2.VideoCapture(0)

# Read the graph.
with tf.gfile.FastGFile('/Users/heizer/github_repos/openCV_adventures/ssd_mobilenet/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    ret, frame = cap.read()

    frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)

    # Read and preprocess an image.
    rows = frame.shape[0]
    cols = frame.shape[1]
    inp = cv2.resize(frame, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    
    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

    cv2.imshow("Image Capture", frame)

cv2.waitKey(0)
cap.release 
cv2.detroyAllWindows()