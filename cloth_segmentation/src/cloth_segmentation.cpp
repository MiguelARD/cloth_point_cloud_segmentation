// ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

// Point Cloud Library (PCL)
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/filter.h>

// Messages 
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <darknet_ros_msgs/BoundingBoxes.h>


// Declare publishers
ros::Publisher pub_cloud;
ros::Publisher pub_img;

// Declare bounding boxes and cloth pixels pointers
darknet_ros_msgs::BoundingBoxesPtr bounding_boxes (new darknet_ros_msgs::BoundingBoxes);
std::vector<cv::Point>* cloth_pixels (new std::vector<cv::Point>());


// Bounding boxes callback
void boxesCallback(const darknet_ros_msgs::BoundingBoxesConstPtr& boxes){
 
  ROS_INFO("Bounding boxes received");

  // Update bounding boxes
  *bounding_boxes = *boxes; 
}


// Color image callback
void imageCallback(const sensor_msgs::ImageConstPtr& image){

  ROS_INFO("Image received");

  try{

  // Number of objects detected
  int num_cloth = bounding_boxes->bounding_boxes.size();
  if (num_cloth==0){return;}

  // Convert ROS image to OpenCV
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Initialize GrabCut mask
  cv::Mat grab_segmentation (cv_ptr->image.size(), CV_8UC1, cv::Scalar(0));
  std::vector<cv::Point> ellipse_centers;
  std::vector<cv::Size> ellipse_axis;

  // Initialize smallest dimensions
  int min_width = image->width;
  int min_height = image->height;

  for (int k=0; k<num_cloth; k++){ // Iterate over bounding boxes

    // Get bounding box coordinates
    cv::Rect rect;
    if (bounding_boxes->bounding_boxes[k].xmin > 0){rect.x = bounding_boxes->bounding_boxes[k].xmin;}
    else{rect.x = 0;}
    if (bounding_boxes->bounding_boxes[k].ymin > 0){rect.y = bounding_boxes->bounding_boxes[k].ymin;}
    else{rect.y = 0;}

    rect.width  = bounding_boxes->bounding_boxes[k].xmax-rect.x;
    rect.height = bounding_boxes->bounding_boxes[k].ymax-rect.y;

    // Increase bounding box
    double box_coeff=0.05;  
    if (rect.x-rect.width*box_coeff/2.0 > 0){rect.x = rect.x-rect.width*box_coeff/2.0;}
    else{rect.x = 0;}
    if (rect.y-rect.height*box_coeff/2.0 > 0){rect.y = rect.y-rect.height*box_coeff/2.0;}
    else{rect.y = 0;}
    rect.width  *= 1.0+box_coeff;
    rect.height *= 1.0+box_coeff;

    // Fill GrabCut mask with bounding boxes:
    //   - Inside, probably foreground
    //   - Outside, definetively background
    cv::rectangle(grab_segmentation, rect, cv::Scalar(3), cv::FILLED, cv::LINE_8);  

    // Get centered ellipse
    double ellipse_coeff=0.5; 
    cv::Size  axis(rect.width*ellipse_coeff, rect.height*ellipse_coeff);
    cv::Point center(rect.x+rect.width/2,rect.y+rect.height/2);

    ellipse_axis.push_back(axis);
    ellipse_centers.push_back(center);

    // Update smallest dimensions
    if (rect.width<min_width){min_width=rect.width;}
    if (rect.height<min_height){min_height=rect.height;}
  }

  for (int k=0; k<num_cloth; k++){
    // Fill GrabCut mask with centered ellipses:
    //   - Inside, definetively foreground
    cv::ellipse(grab_segmentation, ellipse_centers[k], ellipse_axis[k], 0.0, 0.0, 0.0, cv::Scalar(1), cv::FILLED, cv::LINE_8); 
  }

  // Apply GrabCut
  cv::Mat bgModel, fgModel;
  cv::grabCut(cv_ptr->image, grab_segmentation, cv::Rect(), bgModel, fgModel, 1, cv::GC_INIT_WITH_MASK);

  // Filter GrabCut mask with opening operation
  double kernel_coeff = 0.08;
  cv::Mat opening_kernel = cv::getStructuringElement(1, cv::Size(min_width*kernel_coeff,min_height*kernel_coeff));
  cv::morphologyEx(grab_segmentation, grab_segmentation, cv::MORPH_OPEN, opening_kernel);

  // Update cloth pixels
  cv::compare(grab_segmentation, cv::GC_PR_FGD, grab_segmentation, cv::CMP_EQ);
  cv::findNonZero(grab_segmentation, *cloth_pixels);
  
  // Generate and publish segmented cloth color image
  cv::Mat foreground (cv_ptr->image.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv_ptr->image.copyTo(foreground, grab_segmentation);
  *cv_ptr = cv_bridge::CvImage(image->header, sensor_msgs::image_encodings::BGR8, foreground);
  pub_img.publish(cv_ptr->toImageMsg());

  }

  catch(...){}
}


// Auxiliary function for sorting region-growing-based segmentation clusters
bool sortFunction(const pcl::PointIndices& a,const pcl::PointIndices& b) { 
  return (a.indices.size()>b.indices.size()); 
}


// Point cloud callback
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& pointcloud)
{

  ROS_INFO("Point cloud received");

  try{

  // Convert cloud pixels to point cloud indexes
  std::vector<int> cloth_indexes;
  for (int i=0; i<(*cloth_pixels).size(); i++){
    cloth_indexes.push_back((*cloth_pixels)[i].x+(*cloth_pixels)[i].y*pointcloud->width);
  }

  // Convert ROS point cloud to PCL 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*pointcloud, *pcl_cloud);

  // Initialize segmented point cloud container
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Number of objects detected
  int num_cloth = bounding_boxes->bounding_boxes.size();
  if (num_cloth==0){return;}

  // Filter point cloud by color image cloth segmentation indexes
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloth_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*pcl_cloud, cloth_indexes, *cloth_cloud);

  // Remove NAN points
  pcl::removeNaNFromPointCloud(*cloth_cloud,*cloth_cloud,cloth_indexes);

  // Color-based region growing segmentation
  double distance_thresh = 1;             // Distance threshold for cluster
  double angle_thresh_incluster = 20;     // Maximum color angle difference within the cluster
  double angle_thresh_mergecluster = 8;   // Maximum color angle difference for merging clusters
  double min_cluster_size_coeff = 0.15;   // Mimimum cluster size coefficient

  pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  reg.setInputCloud(cloth_cloud);
  reg.setDistanceThreshold(distance_thresh);         
  reg.setPointColorThreshold(angle_thresh_incluster);         
  reg.setRegionColorThreshold(angle_thresh_mergecluster);      
  reg.setMinClusterSize(cloth_cloud->size()*min_cluster_size_coeff);

  // Get region growing clusters
  std::vector <pcl::PointIndices> clusters;
  reg.extract(clusters);

  // Get largest clusters if there is more than one
  if (clusters.size()>1){

    // Initialize auxiliary cloud container
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_aux (new pcl::PointCloud<pcl::PointXYZRGB>);

    // Sort clusters by size
    std::sort(clusters.begin(), clusters.end(), sortFunction);

    int add_n_clusters;
    if (clusters.size()>num_cloth){add_n_clusters=num_cloth;}
    else {add_n_clusters=clusters.size();}

    for (int i=0; i<add_n_clusters; i++){

      // Extract i-th largest cluster
      pcl::copyPointCloud(*cloth_cloud, clusters[i].indices, *cloud_aux);

      // Add cluster to output cloud
      *output_cloud += *cloud_aux;
    }
  }

  else{
    *output_cloud = *cloth_cloud;
  }

  // Generate and publish cloth segmented point cloud
  sensor_msgs::PointCloud2Ptr cloth_cloud_ros (new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*output_cloud, *cloth_cloud_ros);
  cloth_cloud_ros->header = pointcloud->header;
  pub_cloud.publish(*cloth_cloud_ros);

  }

  catch(...){}
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "cloth_segmentation");
  ros::NodeHandle nh;

  // Initialize subscribers
  std::string image_topic, cloud_topic;
  nh.getParam("/cloth_segmentation/subscribers/rgb_reading/topic", image_topic);
  nh.getParam("/cloth_segmentation/subscribers/point_cloud_reading/topic", cloud_topic);

  ros::Subscriber image_sub = nh.subscribe(image_topic, 1, imageCallback);
  ros::Subscriber cloud_sub = nh.subscribe(cloud_topic, 1, cloudCallback);
  ros::Subscriber boxes_sub = nh.subscribe("/darknet_ros/bounding_boxes", 1, boxesCallback);

  // Initialize publishers
  pub_cloud = nh.advertise<sensor_msgs::PointCloud2> ("cloth_segmentation/cloth_pointcloud", 1);
  pub_img   = nh.advertise<sensor_msgs::Image>       ("cloth_segmentation/cloth_image", 1);

  ros::spin();

  return 0;
}
