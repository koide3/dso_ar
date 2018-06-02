import cv2
import numpy
import itertools
import pyquaternion

import rospy
import cv_bridge
import tf_conversions
import message_filters
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *


class Plane:
	def __init__(self, coeffs):
		self.coeffs = numpy.float32(coeffs[:4])
		self.scale = coeffs[4]
		self.centroid = numpy.float32(coeffs[5:])

		self.z_axis = self.coeffs[:3]
		self.x_axis = numpy.cross(self.z_axis, numpy.float32((0, 0, 1)))
		self.y_axis = numpy.cross(self.x_axis, self.z_axis)

		self.x_axis = self.x_axis / numpy.linalg.norm(self.x_axis) * self.scale
		self.y_axis = self.y_axis / numpy.linalg.norm(self.y_axis) * self.scale
		self.z_axis = self.z_axis / numpy.linalg.norm(self.z_axis) * self.scale

		self.vertices = []
		for i in range(-1, 2, 1):
			self.vertices.append(self.centroid + self.x_axis + self.y_axis * i)
			self.vertices.append(self.centroid - self.x_axis + self.y_axis * i)
			self.vertices.append(self.centroid + self.x_axis * i + self.y_axis)
			self.vertices.append(self.centroid + self.x_axis * i - self.y_axis)
		self.vertices = numpy.float32(self.vertices)

	def draw(self, canvas, camera_matrix, distortion, rvec, tvec):
		projected, jacob = cv2.projectPoints(self.vertices, rvec, tvec, camera_matrix, distortion)
		for i in range(0, len(projected), 2):
			pt1 = tuple(projected[i].flatten().astype(numpy.int32))
			pt2 = tuple(projected[i+1].flatten().astype(numpy.int32))
			cv2.line(canvas, pt1, pt2, (0, 0, 255), 2)


class Cube:
	def __init__(self, scale):
		self.scale = scale
		self.vertices = numpy.int32(list(itertools.product((0, 1), (0, 1), (0, 1))))
		ivertices = enumerate(self.vertices)
		self.indices = [(i1, i2) for (i1, x1), (i2, x2) in itertools.combinations(ivertices, 2) if sum(abs(x1 - x2)) == 1]
		self.vertices = (self.vertices.astype(numpy.float32) - (0.5, 0.5, 0.5)) * self.scale

	def draw(self, canvas, camera_matrix, distortion, rvec, tvec, plane):
		vertices = []
		for vertex in self.vertices:
			vertices.append(plane.centroid + plane.z_axis * self.scale / 2)
			vertices[-1] += plane.x_axis * vertex[0] + plane.y_axis * vertex[1] + plane.z_axis * vertex[2]
		vertices = numpy.float32(vertices)

		projected, jacob = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, distortion)

		for i1, i2 in self.indices:
			pt1 = tuple(projected[i1].flatten().astype(numpy.int32))
			pt2 = tuple(projected[i2].flatten().astype(numpy.int32))
			cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)


class SimpleARNode:
	def __init__(self):
		self.canvas = numpy.zeros((480, 640, 4), dtype=numpy.uint8)
		self.image_pub = rospy.Publisher('/ar_image', Image, queue_size=5)

		self.cv_bridge = cv_bridge.CvBridge()
		self.cubes = [Cube(0.5)]
		self.plane = None

		subs = [
			message_filters.Subscriber('/usb_cam_node/camera_info', CameraInfo),
			message_filters.Subscriber('/usb_cam_node/image_raw', Image),
			message_filters.Subscriber('/vodom', PoseStamped)
		]
		self.sync = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.1)
		self.sync.registerCallback(self.callback)

		self.plane_sub = rospy.Subscriber('/plane_coeffs', Float32MultiArray, self.plane_callback)

	def plane_callback(self, plane_msg):
		self.plane = Plane(plane_msg.data)

	def callback(self, camera_info_msg, image_msg, camera_pose_msg):
		if self.plane is None:
			return

		camera_matrix = numpy.float32(camera_info_msg.K).reshape(3, 3)
		distortion = numpy.float32(camera_info_msg.D).flatten()

		frame = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')

		cam2world = tf_conversions.toMatrix(tf_conversions.fromMsg(camera_pose_msg.pose))
		world2cam = numpy.linalg.inv(cam2world)

		tvec = world2cam[:3, 3]
		rvec, jacob = cv2.Rodrigues(world2cam[:3, :3])

		self.plane.draw(frame, camera_matrix, distortion, rvec, tvec)

		for cube in self.cubes:
			cube.draw(frame, camera_matrix, distortion, rvec, tvec, self.plane)

		self.canvas = frame

	def show(self):
		cv2.imshow('canvas', self.canvas)
		cv2.waitKey(10)


def main():
	rospy.init_node('simple_ar')
	node = SimpleARNode()

	while not rospy.is_shutdown():
		node.show()

if __name__ == '__main__':
	main()
