#include <mutex>
#include <iostream>
#include <pcl/common/pca.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>


/** 
 * @brief Palane parameters estimation node
 * This node accumulates input point clouds and applies RANSAC to it.
 * Parameters to be estimates are [plane.x, plane.y, plane.z, plane.w, scale, centroid.x, centroid.y, centroid.z]
 * This implementation has a lot of ad hoc parameters. It may not be able to apply it to other datasets.
 */
class PlaneEstimationNode {
public:
  PlaneEstimationNode()
    : nh(),
      inliers_pub(nh.advertise<sensor_msgs::PointCloud2>("/inlier_points", 10)),
      filtered_pub(nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 10)),
      plane_coeffs_pub(nh.advertise<std_msgs::Float32MultiArray>("/plane_coeffs", 10)),
      points_sub(nh.subscribe("/points", 10, &PlaneEstimationNode::points_callback, this)),
      accumulated(new pcl::PointCloud<pcl::PointXYZ>())
  {

  }

  void points_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& points) {
    if(plane_coeffs_msg) {
      plane_coeffs_pub.publish(plane_coeffs_msg);
      return;
    }

    std::lock_guard<std::mutex> lock(accumulated_mutex);
    std::copy(points->begin(), points->end(), std::back_inserter(accumulated->points));
    accumulated->width = accumulated->size();
    accumulated->height = 1;
    accumulated->is_dense = false;

    std::cout << "estimating plane..." << std::endl;
    estimate_plane(accumulated);
    std::cout << "done" << std::endl;
  }

  /**
   * @brief estimates the plane parameters by applying RANSAC
   */
  void estimate_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    auto filtered = normal_filter(cloud);
    filtered = outlier_removal(filtered);
    if(filtered_pub.getNumSubscribers()){
      filtered->header.frame_id = "world";
      filtered_pub.publish(filtered);
    }

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr indices (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.05);

    seg.setInputCloud (filtered);
    seg.segment(*indices, *coefficients);

    pcl::PointCloud<pcl::PointXYZ>::Ptr inliers(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud<pcl::PointXYZ>(*filtered, indices->indices, *inliers);

    if(inliers->empty()){
      return;
    }

    if(inliers_pub.getNumSubscribers()){
      inliers->header.frame_id = "world";
      inliers_pub.publish(inliers);
    }

    if(plane_coeffs_pub.getNumSubscribers()) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*inliers, centroid);

      // estimates the plane scale based on the distance between each point and the centroid
      std::vector<float> dists(inliers->size());
      std::transform(inliers->begin(), inliers->end(), dists.begin(), [&](const pcl::PointXYZ& pt) {return (pt.getVector4fMap() - centroid).head<3>().norm();});
      std::sort(dists.begin(), dists.end());
      double scale = dists[dists.size() * 0.8];		// eliminate outliers

      std_msgs::Float32MultiArrayPtr coeffs_msg(new std_msgs::Float32MultiArray());
      coeffs_msg->data.resize(coefficients->values.size());
      std::copy(coefficients->values.begin(), coefficients->values.end(), coeffs_msg->data.begin());
      coeffs_msg->data.push_back(scale);
      coeffs_msg->data.push_back(centroid[0]);
      coeffs_msg->data.push_back(centroid[1]);
      coeffs_msg->data.push_back(centroid[2]);
      plane_coeffs_pub.publish(coeffs_msg);

      if(inliers->size() > 3000) {
        plane_coeffs_msg = coeffs_msg;
      }
    }
  }

  /**
   * @brief filters out points with non-vertical normals (here, vertical is (0, 0, -1))
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr normal_filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.03);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute (*normals);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    filtered->reserve(cloud->size());

    Eigen::Vector3f direction(0.0f, 0.0f, -1.0f);
    double thresh = 0.4;
    for(int i=0; i<cloud->size(); i++) {
      const auto& normal = normals->at(i);

      if(!normal.getNormalVector3fMap().array().isNaN().any() &&
         normal.getNormalVector3fMap().dot(direction) > thresh)
      {
        const auto& point = cloud->at(i);
        filtered->push_back(point);
      }
    }

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    return filtered;
  }

  /**
   * @brief filters out outliers with statistical outlier removal
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_removal(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK (20);
    sor.setStddevMulThresh (1.0);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    sor.filter(*filtered);

    return filtered;
  }

private:
  ros::NodeHandle nh;
  ros::Publisher inliers_pub;
  ros::Publisher filtered_pub;
  ros::Publisher plane_coeffs_pub;
  ros::Subscriber points_sub;

  std::mutex accumulated_mutex;
  pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated;

  std_msgs::Float32MultiArrayPtr plane_coeffs_msg;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "plane_estimation_node");
  PlaneEstimationNode node;
  ros::spin();
	return 0;
}
