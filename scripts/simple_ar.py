import cv2
import numpy
import itertools
import pyquaternion

import rospy
import cv_bridge
import tf_conversions
import message_filters
from sensor_msgs.msg import *
from geometry_msgs.msg import *


class Cube:
	def __init__(self, pos, size):
		self.pos = numpy.float32(pos)

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


class ScaleEstimater:
	def __init__(self):
		self.tvecs = []

	def estimate(self, tvec):
		if len(self.tvecs) < 100:
			self.tvecs.append(tvec)
			return False, None

		tmin = numpy.min(self.tvecs[20:], axis=0)
		tmax = numpy.max(self.tvecs[20:], axis=0)

		return True, numpy.linalg.norm(tmax - tmin)


class SimpleARNode:
	def __init__(self):
		subs = [
			message_filters.Subscriber('/usb_cam_node/camera_info', CameraInfo),
			message_filters.Subscriber('/usb_cam_node/image_raw', Image),
			message_filters.Subscriber('/vodom', PoseStamped)
		]
		self.sync = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.1)
		self.sync.registerCallback(self.callback)

		self.image_pub = rospy.Publisher('/ar_image', Image, queue_size=5)

		self.cv_bridge = cv_bridge.CvBridge()
		self.scale_estimater = ScaleEstimater()
		self.cubes = []
		self.frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)

	def put_cube(self, cam2world):
		pos = cam2world.dot(numpy.float32((0, 0, 0.5, 1)))
		self.cubes.append(Cube(pos[:3], 0.15))

	def callback(self, camera_info_msg, image_msg, camera_pose_msg):
		camera_matrix = numpy.float32(camera_info_msg.K).reshape(3, 3)
		distortion = numpy.float32(camera_info_msg.D).flatten()

		frame = self.cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')

		cam2world = tf_conversions.toMatrix(tf_conversions.fromMsg(camera_pose_msg.pose))
		world2cam = numpy.linalg.inv(cam2world)

		tvec = world2cam[:3, 3]
		rvec, jacob = cv2.Rodrigues(world2cam[:3, :3])

		ret, scale = self.scale_estimater.estimate(tvec)
		if not ret:
			cv2.putText(frame, 'estimating scale...', (10, 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
		else:
			scale = scale / 0.2
			for cube in self.cubes:
				cube.draw(frame, camera_matrix, distortion, rvec, tvec, scale)

		self.frame = frame
		self.cam2world = cam2world

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
