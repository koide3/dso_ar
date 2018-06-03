#!/usr/bin/python
import cv2
import numpy
import itertools

import rospy
import cv_bridge
import tf_conversions
import message_filters
from sensor_msgs.msg import *
from geometry_msgs.msg import *


# A class for displaying a cube
class Cube:
	def __init__(self, pos, size):
		self.pos = numpy.float32(pos)

		# generates 3D vertices and indices which connect close vertices (distance == 1)
		self.vertices = numpy.int32(list(itertools.product((0, 1), (0, 1), (0, 1))))
		ivertices = enumerate(self.vertices)
		self.indices = [(i1, i2) for (i1, x1), (i2, x2) in itertools.combinations(ivertices, 2) if sum(abs(x1 - x2)) == 1]
		self.vertices = (self.vertices.astype(numpy.float32) - (0.5, 0.5, 0.5)) * size + self.pos

	def draw(self, canvas, camera_matrix, distortion, rvec, tvec, scale):
		projected, jacob = cv2.projectPoints(self.vertices * scale, rvec, tvec, camera_matrix, distortion)

		for i1, i2 in self.indices:
			pt1 = tuple(projected[i1].flatten().astype(numpy.int32))
			pt2 = tuple(projected[i2].flatten().astype(numpy.int32))
			cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)


# A class to estimate the scale of the environment
# It calculates the size of the area where the camera moved in the first 100 frames as the scale unit
class ScaleEstimater:
	def __init__(self):
		self.tvecs = []

	# returns (False, None) if the number of acquired tvecs is insufficient otherwise returns (True, scale)
	def estimate(self, tvec):
		if len(self.tvecs) < 100:
			self.tvecs.append(tvec)
			return False, None

		tmin = numpy.min(self.tvecs[20:], axis=0)
		tmax = numpy.max(self.tvecs[20:], axis=0)

		return True, numpy.linalg.norm(tmax - tmin)


# A ROS node for simple AR with visual odometry
# By pressing the space button, you can put boxes in the AR environment
class SimpleARNode:
	def __init__(self):
		self.cubes = []
		self.scale_estimater = ScaleEstimater()
		self.frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)

		self.cv_bridge = cv_bridge.CvBridge()

		subs = [
			message_filters.Subscriber('/usb_cam_node/camera_info', CameraInfo),
			message_filters.Subscriber('/usb_cam_node/image_raw', Image),
			message_filters.Subscriber('/vodom', PoseStamped)
		]
		self.sync = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.1)
		self.sync.registerCallback(self.callback)

	# puts a cube in front of the camera
	def put_cube(self, cam2world):
		pos = cam2world.dot(numpy.float32((0, 0, 0.5, 1)))
		self.cubes.append(Cube(pos[:3], 0.15))

	# image and camera pose callback
	def callback(self, camera_info_msg, image_msg, camera_pose_msg):
		# camera parameters and image
		camera_matrix = numpy.float32(camera_info_msg.K).reshape(3, 3)
		distortion = numpy.float32(camera_info_msg.D).flatten()
		frame = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')

		# camera pose
		cam2world = tf_conversions.toMatrix(tf_conversions.fromMsg(camera_pose_msg.pose))
		world2cam = numpy.linalg.inv(cam2world)

		tvec = world2cam[:3, 3]
		rvec, jacob = cv2.Rodrigues(world2cam[:3, :3])

		# estimates the scale of the environment
		ret, scale = self.scale_estimater.estimate(tvec)
		if not ret:
			cv2.putText(frame, 'estimating scale...', (10, 30), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 255, 255))
		else:
			# assuming that the camera moves 20cm in the first 100 frames
			scale = scale / 0.2
			for cube in self.cubes:
				cube.draw(frame, camera_matrix, distortion, rvec, tvec, scale)

		self.frame = frame
		self.cam2world = cam2world

	# shows the superimposed image!! and puts a cube!!
	def show(self):
		cv2.imshow('frame', self.frame)
		if cv2.waitKey(10) == ord(' '):
			self.put_cube(self.cam2world)


def main():
	rospy.init_node('simple_ar')
	node = SimpleARNode()
	while not rospy.is_shutdown():
		node.show()

if __name__ == '__main__':
	main()
