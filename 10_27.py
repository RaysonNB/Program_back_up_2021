#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image  # sensor_msgs/Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np

path ="/home/robot/Desktop/aikit_lib/haarcascade_frontalface_default.xml"
model = cv2.CascadeClassifier(path)
print 'hi'

#camera = cv2.VideoCapture(0)
def callback_image(msg):
	global image
	image = CvBridge().imgmsg_to_cv2(msg,"bgr8")


def callback_depth(msg):
	global depth
	tmp = CvBridge().imgmsg_to_cv2(msg,"passthrough")
	depth = np.array(tmp,dtype=np.float32)


if __name__ == "__main__":
	rospy.init_node("a10_27")
	rospy.loginfo("demo start!")
	image = None
	depth = None
	rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
	cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
	msg = Twist()
	#image=rospy.wait_for_message("/camera/rgb/image_raw", Image)
	#rospy.wait_for_message("/camera/rgb/image_raw". Image)
	rospy.loginfo("kennygay")
	#frame = image
	topic_name = "/camera/depth/image_raw"
	rospy.Subscriber(topic_name, Image, callback_depth)
	#rospy.wait_for_message(topic_name,Image)

	# d = depth[y][x]
	while not rospy.is_shutdown():
		rospy.Rate(20).sleep()
		#print image.shape
		color = (0, 255, 0)
		if image is None: continue
		if depth is None: continue
		old_h,old_w,c = image.shape
		outputs = model.detectMultiScale(image)
		gray = depth/ np.max(depth)
		target = None
		for box in outputs:
			x, y, w, h = box
			in_x=x+(w//2)
			in_y=y+(h//2)
			if target is None:
				target = box 
			if w * h > target[2] * target[3]:
				target = box 

			cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
			#cv2.putText(image, gray, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.circle(image,(in_x,in_y), 2, color, 2)
			new_x = x + w // 2
			new_y = y + h // 2

		angular_z = 0
		if target is not None:
			error = float(320 - in_x)
			#print "distance:", depth[y][x]#, "error", error
			if abs(error) > 20:
				angular_z = error / 320 * 0.4
			else:
				msg.linear.x=0.0
				t0 = rospy.Time.now().to_sec()
				while depth[y][x] > 45:
					msg.linear.x=0.3
					cmd_vel.publish(msg)
					
		#print angular_z
		cmd_msg = Twist()
		cmd_msg.angular.z = angular_z
		cmd_vel.publish(cmd_msg)
			
		cv2.imshow("image",image)
		key_code = cv2.waitKey(1)
		if key_code in [27, ord('q')]:
		    break
	cv2.destroyAllWindows()
	rospy.loginfo("demo end!")

