#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>
#include <math.h>
#include <random>

using namespace std;

pcl::search::KdTree<pcl::PointXYZ> kdtreeLocalMap;
vector<int> pointIdxRadiusSearch;
vector<float> pointRadiusSquaredDistance;

random_device rd;
default_random_engine eng(rd());
uniform_real_distribution<double> rand_x;
uniform_real_distribution<double> rand_y;
uniform_real_distribution<double> rand_w;
uniform_real_distribution<double> rand_h;

ros::Publisher _local_map_pub;
ros::Publisher _all_map_pub;
ros::Subscriber _odom_sub;

vector<double> _state;

int _obs_num;
double _x_size, _y_size, _z_size;
double _x_l, _x_h, _y_l, _y_h, _w_l, _w_h, _h_l, _h_h;
double _z_limit, _sensing_range, _resolution, _sense_rate, _init_x, _init_y;

bool _map_ok = false;
bool _has_odom = false;

sensor_msgs::PointCloud2 localMap_pcd;
sensor_msgs::PointCloud2 globalMap_pcd;
pcl::PointCloud<pcl::PointXYZ> cloudMap;

void RandomMapGenerate()
{
    pcl::PointXYZ pt_random;

    rand_x = uniform_real_distribution<double>(_x_l, _x_h);
    rand_y = uniform_real_distribution<double>(_y_l, _y_h);
    rand_w = uniform_real_distribution<double>(_w_l, _w_h);
    rand_h = uniform_real_distribution<double>(_h_l, _h_h);

    for (int i = 0; i < _obs_num; i++)
    {
        double x, y, w, h;
        x = rand_x(eng);
        y = rand_y(eng);
        w = rand_w(eng);

        if (sqrt(pow(x - _init_x, 2) + pow(y - _init_y, 2)) < 2.0)
            continue;

        x = floor(x / _resolution) * _resolution + _resolution / 2.0;   // 把每个随机点放入grid中心
        y = floor(y / _resolution) * _resolution + _resolution / 2.0;

        int widNum = ceil(w / _resolution);

        for (int r = -widNum / 2.0; r < widNum / 2.0; r++)
            for (int s = -widNum / 2.0; s < widNum / 2.0; s++)
            {
                h = rand_h(eng);
                int heiNum = ceil(h / _resolution);
                for (int t = 0; t < heiNum; t++)
                {
                    pt_random.x = x + (r + 0.5) * _resolution;   // *0.5也是为了将点放入grid中心
                    pt_random.y = y + (s + 0.5) * _resolution;
                    pt_random.z = (t + 0.5) * _resolution;
                    cloudMap.points.push_back(pt_random);
                }
            }
    }

    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;

    ROS_WARN("Finished generate random map");

    kdtreeLocalMap.setInputCloud(cloudMap.makeShared());

    _map_ok = true;
}

void rcvOdometryCallbck(const nav_msgs::Odometry odom)
{
    if (odom.child_frame_id == "X" || odom.child_frame_id == "O")
        return;
    _has_odom = true;

    _state = {
        odom.pose.pose.position.x,
        odom.pose.pose.position.y,
        odom.pose.pose.position.z,
        odom.twist.twist.linear.x,
        odom.twist.twist.linear.y,
        odom.twist.twist.linear.z,
        0.0, 0.0, 0.0};
}

int i = 0;
void pubSensedPoints()  // default:10HZ
{
    if (i < 10)
    {
        pcl::toROSMsg(cloudMap, globalMap_pcd);
        globalMap_pcd.header.frame_id = "world";
        _all_map_pub.publish(globalMap_pcd);     // 前1s发布global map
    }

    i++;
    if (!_map_ok || !_has_odom)
        return;

    pcl::PointCloud<pcl::PointXYZ> localMap;

    pcl::PointXYZ searchPoint(_state[0], _state[1], _state[2]);   // 以当前里程计信息为中心
    pointIdxRadiusSearch.clear();
    pointRadiusSquaredDistance.clear();

    pcl::PointXYZ pt;

    if (isnan(searchPoint.x) || isnan(searchPoint.y) || isnan(searchPoint.z))
        return;

    // 通过kd-tree搜索近邻, _sensing_range为搜索半径, 将所有半径范围内的邻居信息存入pointIdxRadiusSearch和pointRadiusSquaredDistance.
    // reference : https://www.cnblogs.com/li-yao7758258/p/6437440.html
    if (kdtreeLocalMap.radiusSearch(searchPoint, _sensing_range, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
        {
            pt = cloudMap.points[pointIdxRadiusSearch[i]];
            localMap.points.push_back(pt);
        }
    }
    else
    {
        ROS_ERROR("[Map server] No obstacles in local map.");
        return;
    }

    // reference : https://blog.csdn.net/Linear_Luo/article/details/52647790
    localMap.width = localMap.points.size();    // 对于无组织或者说无结构的点云来说，width就是指点云中点的个数. 对于有结构的点云来说，width指点云数据集一行中点的个数
    localMap.height = 1;    // 对于无结构的点云来说，height值为1. 对于有结构点云来说，height代表点云的总行数
    localMap.is_dense = true;   // true:points中的数据是有限的, false:点云中的点包含 Inf/NaN 这种值

    pcl::toROSMsg(localMap, localMap_pcd);
    localMap_pcd.header.frame_id = "world";
    _local_map_pub.publish(localMap_pcd);    // 每个感知周期发布一次local map
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "random_map_sensing");
    ros::NodeHandle n("~");

    _local_map_pub = n.advertise<sensor_msgs::PointCloud2>("random_forest", 1);
    _all_map_pub = n.advertise<sensor_msgs::PointCloud2>("all_map", 1);

    _odom_sub = n.subscribe("odometry", 50, rcvOdometryCallbck);

    n.param("init_state_x", _init_x, 0.0);
    n.param("init_state_y", _init_y, 0.0);

    n.param("map/x_size", _x_size, 50.0);
    n.param("map/y_size", _y_size, 50.0);
    n.param("map/z_size", _z_size, 5.0);
    n.param("map/obs_num", _obs_num, 30);
    n.param("map/resolution", _resolution, 0.2);

    n.param("ObstacleShape/lower_rad", _w_l, 0.3);
    n.param("ObstacleShape/upper_rad", _w_h, 0.8);
    n.param("ObstacleShape/lower_hei", _h_l, 3.0);
    n.param("ObstacleShape/upper_hei", _h_h, 7.0);

    n.param("sensing/radius", _sensing_range, 10.0);
    n.param("sensing/radius", _sense_rate, 10.0);

    _x_l = -_x_size / 2.0;
    _x_h = +_x_size / 2.0;

    _y_l = -_y_size / 2.0;
    _y_h = +_y_size / 2.0;

    _obs_num = min(_obs_num, (int)_x_size * 10);
    _z_limit = _z_size;

    RandomMapGenerate();

    ros::Rate loop_rate(_sense_rate);

    while (ros::ok())
    {
        pubSensedPoints();
        ros::spinOnce();
        loop_rate.sleep();
    }
}