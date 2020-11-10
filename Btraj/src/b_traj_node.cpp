#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "trajectory_generator.h"
#include "bezier_base.h"
#include "data_type.h"
#include "utils.h"
#include "a_star.h"
#include "backward.hpp"

#include "quadrotor_msgs/PositionCommand.h"
#include "quadrotor_msgs/PolynomialTrajectory.h"

using namespace std;
using namespace Eigen;
using namespace sdf_tools;

namespace backward {
backward::SignalHandling sh;
}

// simulation param from launch file
double _vis_traj_width;
double _resolution, _inv_resolution;
double _cloud_margin, _cube_margin, _check_horizon, _stop_horizon;
double _x_size, _y_size, _z_size, _x_local_size, _y_local_size, _z_local_size;    
double _MAX_Vel, _MAX_Acc;
bool   _is_use_fm, _is_proj_cube, _is_limit_vel, _is_limit_acc;
int    _step_length, _max_inflate_iter, _traj_order;
double _minimize_order;

// useful global variables
nav_msgs::Odometry _odom;
bool _has_odom  = false;
bool _has_map   = false;
bool _has_target= false;
bool _has_traj  = false;
bool _is_emerg  = false;
bool _is_init   = true;

Vector3d _start_pt, _start_vel, _start_acc, _end_pt;
double _init_x, _init_y, _init_z;
Vector3d _map_origin;
double _pt_max_x, _pt_min_x, _pt_max_y, _pt_min_y, _pt_max_z, _pt_min_z;
int _max_x_id, _max_y_id, _max_z_id, _max_local_x_id, _max_local_y_id, _max_local_z_id;
int _traj_id = 1;
COLLISION_CELL _free_cell(0.0);
COLLISION_CELL _obst_cell(1.0);
// ros related
ros::Subscriber _map_sub, _pts_sub, _odom_sub;
ros::Publisher _fm_path_vis_pub, _local_map_vis_pub, _inf_map_vis_pub, _corridor_vis_pub, _traj_vis_pub, _grid_path_vis_pub, _nodes_vis_pub, _traj_pub, _checkTraj_vis_pub, _stopTraj_vis_pub;

// trajectory related
int _seg_num;
VectorXd _seg_time;
MatrixXd _bezier_coeff;

// bezier basis constant
MatrixXd _MQM, _FM;
VectorXd _C, _Cv, _Ca, _Cj;

// useful object
quadrotor_msgs::PolynomialTrajectory _traj;
ros::Time _start_time = ros::TIME_MAX;
TrajectoryGenerator _trajectoryGenerator;
CollisionMapGrid * collision_map       = new CollisionMapGrid();
CollisionMapGrid * collision_map_local = new CollisionMapGrid();
gridPathFinder * path_finder           = new gridPathFinder();

void rcvWaypointsCallback(const nav_msgs::Path & wp);
void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map);
void rcvOdometryCallbck(const nav_msgs::Odometry odom);

void trajPlanning();
bool checkExecTraj();
bool checkCoordObs(Vector3d checkPt);
vector<pcl::PointXYZ> pointInflate( pcl::PointXYZ pt);

void visPath(vector<Vector3d> path);
void visCorridor(vector<Cube> corridor);
void visGridPath( vector<Vector3d> grid_path);
void visExpNode( vector<GridNodePtr> nodes);
void visBezierTrajectory(MatrixXd polyCoeff, VectorXd time);

pair<Cube, bool> inflateCube(Cube cube, Cube lstcube);
Cube generateCube( Vector3d pt) ;
bool isContains(Cube cube1, Cube cube2);
void corridorSimplify(vector<Cube> & cubicList);
vector<Cube> corridorGeneration(vector<Vector3d> path_coord, vector<double> time);
vector<Cube> corridorGeneration(vector<Vector3d> path_coord);
void sortPath(vector<Vector3d> & path_coord, vector<double> & time);
void timeAllocation(vector<Cube> & corridor, vector<double> time);
void timeAllocation(vector<Cube> & corridor);

VectorXd getStateFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now );
Vector3d getPosFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now );
quadrotor_msgs::PolynomialTrajectory getBezierTraj();

void rcvOdometryCallbck(const nav_msgs::Odometry odom)
{
    if (odom.header.frame_id != "uav") 
        return ;
    
    _odom = odom;
    _has_odom = true;

    // 每次local_map的原点
    _start_pt(0)  = _odom.pose.pose.position.x;
    _start_pt(1)  = _odom.pose.pose.position.y;
    _start_pt(2)  = _odom.pose.pose.position.z;

    _start_vel(0) = _odom.twist.twist.linear.x;
    _start_vel(1) = _odom.twist.twist.linear.y;
    _start_vel(2) = _odom.twist.twist.linear.z;

    _start_acc(0) = _odom.twist.twist.angular.x;
    _start_acc(1) = _odom.twist.twist.angular.y;
    _start_acc(2) = _odom.twist.twist.angular.z;

    if (std::isnan(_odom.pose.pose.position.x) || std::isnan(_odom.pose.pose.position.y) || std::isnan(_odom.pose.pose.position.z))
        return;

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(_odom.pose.pose.position.x, _odom.pose.pose.position.y, _odom.pose.pose.position.z));
    transform.setRotation(tf::Quaternion(0, 0, 0, 1.0));
    // tf用于rviz显示模型
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "quadrotor"));
}

void rcvWaypointsCallback(const nav_msgs::Path & wp)
{     
    if(wp.poses[0].pose.position.z < 0.0)
        return;

    _is_init = false;
    _end_pt << wp.poses[0].pose.position.x,
               wp.poses[0].pose.position.y,
               wp.poses[0].pose.position.z;

    _has_target = true;
    _is_emerg   = true;

    ROS_INFO("[Fast Marching Node] receive the way-points");

    trajPlanning();    // 第一条轨迹的生成
}

Vector3d _local_origin;
void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 & pointcloud_map)
{   
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(pointcloud_map, cloud);
    
    if((int)cloud.points.size() == 0)
        return;

    delete collision_map_local;

    // ros::Time time_1 = ros::Time::now();
    
    collision_map->RestMap();
    
    // 以右下角为local_map的原点,当前机器人处于local_map的中心位置
    double local_c_x = (int)((_start_pt(0) - _x_local_size/2.0)  * _inv_resolution + 0.5) * _resolution;
    double local_c_y = (int)((_start_pt(1) - _y_local_size/2.0)  * _inv_resolution + 0.5) * _resolution;
    double local_c_z = (int)((_start_pt(2) - _z_local_size/2.0)  * _inv_resolution + 0.5) * _resolution;

    _local_origin << local_c_x, local_c_y, local_c_z;   // 当前local_map的原点

    Translation3d origin_local_translation( _local_origin(0), _local_origin(1), _local_origin(2));
    Quaterniond origin_local_rotation(1.0, 0.0, 0.0, 0.0);

    Affine3d origin_local_transform = origin_local_translation * origin_local_rotation;
    
    double _buffer_size = 2 * _MAX_Vel;    // 速度越大,local_map范围越大?
    double _x_buffer_size = _x_local_size + _buffer_size;   // *_local_size为局部地图的范围
    double _y_buffer_size = _y_local_size + _buffer_size;
    double _z_buffer_size = _z_local_size + _buffer_size;

    collision_map_local = new CollisionMapGrid(origin_local_transform, "world", _resolution, _x_buffer_size, _y_buffer_size, _z_buffer_size, _free_cell);

    vector<pcl::PointXYZ> inflatePts(20);   // 初始化20个元素
    pcl::PointCloud<pcl::PointXYZ> cloud_inflation;
    pcl::PointCloud<pcl::PointXYZ> cloud_local;

    for (int idx = 0; idx < (int)cloud.points.size(); idx++)
    {   
        auto mk = cloud.points[idx];
        pcl::PointXYZ pt(mk.x, mk.y, mk.z);

        // 感知到的地图是通过kd-tree在半径为15的圆内搜索的点,local_map是边长为15的正方形,所以需要裁剪
        if( fabs(pt.x - _start_pt(0)) > _x_local_size / 2.0 || fabs(pt.y - _start_pt(1)) > _y_local_size / 2.0 || fabs(pt.z - _start_pt(2)) > _z_local_size / 2.0 )
            continue; 
        
        cloud_local.push_back(pt);      // 用于显示
        inflatePts = pointInflate(pt);  // 用于显示
        for(int i = 0; i < (int)inflatePts.size(); i++)
        {   
            pcl::PointXYZ inf_pt = inflatePts[i];
            Vector3d addPt(inf_pt.x, inf_pt.y, inf_pt.z);
            collision_map_local->Set3d(addPt, _obst_cell);  // 将膨胀后的地图用作碰撞检测
            collision_map->Set3d(addPt, _obst_cell);
            cloud_inflation.push_back(inf_pt);  // 这里将所有膨胀后的点云都显示出来了,没有剪枝,数据量太大会造成卡顿
        }
    }
    _has_map = true;

    cloud_inflation.width = cloud_inflation.points.size();
    cloud_inflation.height = 1;
    cloud_inflation.is_dense = true;
    cloud_inflation.header.frame_id = "world";

    cloud_local.width = cloud_local.points.size();
    cloud_local.height = 1;
    cloud_local.is_dense = true;
    cloud_local.header.frame_id = "world";

    sensor_msgs::PointCloud2 inflateMap, localMap;
    
    pcl::toROSMsg(cloud_inflation, inflateMap);
    pcl::toROSMsg(cloud_local, localMap);
    _inf_map_vis_pub.publish(inflateMap);
    _local_map_vis_pub.publish(localMap);

    // ros::Time time_3 = ros::Time::now();
    // ROS_WARN("Time in receving the map is %f", (time_3 - time_1).toSec());

    if( checkExecTraj() == true )   // 检查轨迹,若碰到障碍,重新规划
        trajPlanning(); 
}

// 将一个点膨胀成多个点
vector<pcl::PointXYZ> pointInflate( pcl::PointXYZ pt)
{
    int num   = int(_cloud_margin * _inv_resolution);
    int num_z = max(1, num / 2);
    vector<pcl::PointXYZ> infPts(20);
    pcl::PointXYZ pt_inf;

    // 这里效率不高:　相邻的点云分别膨胀后会有重叠的点，降低效率
    for(int x = -num ; x <= num; x ++ )
        for(int y = -num ; y <= num; y ++ )
            for(int z = -num_z ; z <= num_z; z ++ )
            {
                pt_inf.x = pt.x + x * _resolution;
                pt_inf.y = pt.y + y * _resolution;
                pt_inf.z = pt.z + z * _resolution;

                infPts.push_back( pt_inf );
            }

    return infPts;
}

bool checkExecTraj()
{   
    if( _has_traj == false ) 
        return false;

    Vector3d traj_pt;

    visualization_msgs::Marker _check_traj_vis, _stop_traj_vis;

    geometry_msgs::Point pt;
    _stop_traj_vis.header.stamp    = _check_traj_vis.header.stamp    = ros::Time::now();
    _stop_traj_vis.header.frame_id = _check_traj_vis.header.frame_id = "world";
    
    _check_traj_vis.ns = "trajectory/check_trajectory";
    _stop_traj_vis.ns  = "trajectory/stop_trajectory";

    _stop_traj_vis.id     = _check_traj_vis.id = 0;
    _stop_traj_vis.type   = _check_traj_vis.type = visualization_msgs::Marker::SPHERE_LIST;
    _stop_traj_vis.action = _check_traj_vis.action = visualization_msgs::Marker::ADD;

    _stop_traj_vis.scale.x = 2.0 * _vis_traj_width;
    _stop_traj_vis.scale.y = 2.0 * _vis_traj_width;
    _stop_traj_vis.scale.z = 2.0 * _vis_traj_width;

    _check_traj_vis.scale.x = 1.5 * _vis_traj_width;
    _check_traj_vis.scale.y = 1.5 * _vis_traj_width;
    _check_traj_vis.scale.z = 1.5 * _vis_traj_width;

    _check_traj_vis.pose.orientation.x = 0.0;
    _check_traj_vis.pose.orientation.y = 0.0;
    _check_traj_vis.pose.orientation.z = 0.0;
    _check_traj_vis.pose.orientation.w = 1.0;

    _stop_traj_vis.pose = _check_traj_vis.pose;

    _stop_traj_vis.color.r = 0.0;
    _stop_traj_vis.color.g = 1.0;
    _stop_traj_vis.color.b = 0.0;
    _stop_traj_vis.color.a = 1.0;

    _check_traj_vis.color.r = 0.0;
    _check_traj_vis.color.g = 0.0;
    _check_traj_vis.color.b = 1.0;
    _check_traj_vis.color.a = 1.0;

    double t_s = max(0.0, (_odom.header.stamp - _start_time).toSec());      
    int idx;
    for (idx = 0; idx < _seg_num; ++idx)
    {
        if( t_s  > _seg_time(idx) && idx + 1 < _seg_num)
            t_s -= _seg_time(idx);  // idx: 当前帧odom的时间戳所对应的那一段轨迹, t_s: 当前相对于那一段轨迹起点的时间
        else 
            break;
    }

    double duration = 0.0;
    double t_ss;
    for (int i = idx; i < _seg_num; i++) // 检测从当前位置到轨迹末端是否触碰障碍, 同时显示轨迹
    {
        t_ss = (i == idx) ? t_s : 0.0; // 每一段轨迹的起点, 若为第一段轨迹, 从起始时间开始遍历, 若不是当前段轨迹, 则需要完整遍历
        for (double t = t_ss; t < _seg_time(i); t += 0.01)
        {
            double t_d = duration + t - t_ss; // 相对于轨迹起点的delta_t
            if (t_d > _check_horizon) // 超出检查范围, 退出
                break;
            traj_pt = getPosFromBezier(_bezier_coeff, t / _seg_time(i), i); // 这里的t是相对于当前段轨迹起点的时间
            pt.x = traj_pt(0) = _seg_time(i) * traj_pt(0);
            pt.y = traj_pt(1) = _seg_time(i) * traj_pt(1); // _seg_time(i)为比例系数
            pt.z = traj_pt(2) = _seg_time(i) * traj_pt(2);

            _check_traj_vis.points.push_back(pt);

            if (t_d <= _stop_horizon)
                _stop_traj_vis.points.push_back(pt);

            if (checkCoordObs(traj_pt))
            {
                ROS_WARN("predicted collision time is %f ahead", t_d);  // 轨迹上出现障碍, 重新规划

                if (t_d <= _stop_horizon)
                {
                    ROS_ERROR("emergency occurs in time is %f ahead", t_d);  // 坠机
                    _is_emerg = true;
                }

                _checkTraj_vis_pub.publish(_check_traj_vis);
                _stopTraj_vis_pub.publish(_stop_traj_vis);

                return true;
            }
        }
        duration += _seg_time(i) - t_ss; // 每一段的累计时间长度 
    }

    _checkTraj_vis_pub.publish(_check_traj_vis); 
    _stopTraj_vis_pub.publish(_stop_traj_vis); 

    return false;
}

bool checkCoordObs(Vector3d checkPt)
{       
    if(collision_map->Get(checkPt(0), checkPt(1), checkPt(2)).first.occupancy > 0.0 )
        return true;

    return false;
}

pair<Cube, bool> inflateCube(Cube cube, Cube lstcube)
{   
    Cube cubeMax = cube;

    // Inflate sequence: right, left, front, back, above, below                                                                              
    MatrixXi vertex_idx(8, 3);
    
    // 判断当前的路径点是否触碰障碍,因为传入的cube是一个点
    for (int i = 0; i < 8; i++)
    { 
        double coord_x = max(min(cube.vertex(i, 0), _pt_max_x), _pt_min_x);
        double coord_y = max(min(cube.vertex(i, 1), _pt_max_y), _pt_min_y);
        double coord_z = max(min(cube.vertex(i, 2), _pt_max_z), _pt_min_z);
        Vector3d coord(coord_x, coord_y, coord_z);

        Vector3i pt_idx = collision_map->LocationToGridIndex(coord);
        
        if( collision_map->Get( (int64_t)pt_idx(0), (int64_t)pt_idx(1), (int64_t)pt_idx(2) ).first.occupancy > 0.5 )
        {       
            ROS_ERROR("[Planning Node] path has node in obstacles !");
            return make_pair(cubeMax, false);
        }
        
        vertex_idx.row(i) = pt_idx;  // 若未触碰障碍,将该点的x, y, z坐标赋值给vertex_idx的对应行,等待膨胀
    }

    int id_x, id_y, id_z;

    /*
               P4------------P3 
               /|           /|              ^
              / |          / |              | z
            P1--|---------P2 |              |
             |  P8--------|--p7             |
             | /          | /               /--------> y
             |/           |/               /  
            P5------------P6              / x
    */           

    bool collide;

    MatrixXi vertex_idx_lst = vertex_idx;   // 存储未膨胀前cube的idx

    // 依次将cube的某个面(例如P1-P4-P8-P5)向对应的坐标轴方向扩展_step_length, 并检查这个面是否触碰障碍物
    int iter = 0;
    while(iter < _max_inflate_iter) // 迭代次数也就是最大扩展距离
    {   
        // Y Axis
        int y_lo = max(0, vertex_idx(0, 1) - _step_length);
        int y_up = min(_max_y_id, vertex_idx(1, 1) + _step_length);

        // Y+ now is the right side : (p2 -- p3 -- p7 -- p6) face
        // ############################################################################################################
        collide = false;
        for(id_y = vertex_idx(1, 1); id_y <= y_up; id_y++ )
        {   
            if( collide == true) 
                break;
            
            for(id_x = vertex_idx(1, 0); id_x >= vertex_idx(2, 0); id_x-- ) // P2P3
            {
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(1, 2); id_z >= vertex_idx(5, 2); id_z-- ) // P2P6
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(1, 1) = max(id_y-2, vertex_idx(1, 1));   // _step_length = 1, 若有障碍说明之前cube就已经到达边界 
            vertex_idx(2, 1) = max(id_y-2, vertex_idx(2, 1));   // 此时id_y = y_up+1
            vertex_idx(6, 1) = max(id_y-2, vertex_idx(6, 1));   // max函数的意义不明
            vertex_idx(5, 1) = max(id_y-2, vertex_idx(5, 1));
        }
        else
            vertex_idx(1, 1) = vertex_idx(2, 1) = vertex_idx(6, 1) = vertex_idx(5, 1) = id_y - 1;  // for循环后id_y = y_up+1


        // Y- now is the left side : (p1 -- p4 -- p8 -- p5) face 
        // ############################################################################################################
        collide  = false;

        for(id_y = vertex_idx(0, 1); id_y >= y_lo; id_y-- ) 
        {   
            if( collide == true)   // 退出多层for循环
                break;
            
            for(id_x = vertex_idx(0, 0); id_x >= vertex_idx(3, 0); id_x-- ) // P1P4
            {    
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(0, 2); id_z >= vertex_idx(4, 2); id_z-- ) // P1P5
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(0, 1) = min(id_y+2, vertex_idx(0, 1));
            vertex_idx(3, 1) = min(id_y+2, vertex_idx(3, 1));
            vertex_idx(7, 1) = min(id_y+2, vertex_idx(7, 1));
            vertex_idx(4, 1) = min(id_y+2, vertex_idx(4, 1));
        }
        else
            vertex_idx(0, 1) = vertex_idx(3, 1) = vertex_idx(7, 1) = vertex_idx(4, 1) = id_y + 1;   
        


        // X Axis
        int x_lo = max(0, vertex_idx(3, 0) - _step_length);
        int x_up = min(_max_x_id, vertex_idx(0, 0) + _step_length);
        // X + now is the front side : (p1 -- p2 -- p6 -- p5) face
        // ############################################################################################################

        collide = false;
        for(id_x = vertex_idx(0, 0); id_x <= x_up; id_x++ )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(0, 1); id_y <= vertex_idx(1, 1); id_y++ ) // P1P2
            {
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(0, 2); id_z >= vertex_idx(4, 2); id_z-- ) // P1P5
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(0, 0) = max(id_x-2, vertex_idx(0, 0)); 
            vertex_idx(1, 0) = max(id_x-2, vertex_idx(1, 0)); 
            vertex_idx(5, 0) = max(id_x-2, vertex_idx(5, 0)); 
            vertex_idx(4, 0) = max(id_x-2, vertex_idx(4, 0)); 
        }
        else
            vertex_idx(0, 0) = vertex_idx(1, 0) = vertex_idx(5, 0) = vertex_idx(4, 0) = id_x - 1;    

        // X- now is the back side : (p4 -- p3 -- p7 -- p8) face
        // ############################################################################################################
        collide = false;
        for(id_x = vertex_idx(3, 0); id_x >= x_lo; id_x-- )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(3, 1); id_y <= vertex_idx(2, 1); id_y++ ) // P4P3
            {
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(3, 2); id_z >= vertex_idx(7, 2); id_z-- ) //P4P8
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(3, 0) = min(id_x+2, vertex_idx(3, 0)); 
            vertex_idx(2, 0) = min(id_x+2, vertex_idx(2, 0)); 
            vertex_idx(6, 0) = min(id_x+2, vertex_idx(6, 0)); 
            vertex_idx(7, 0) = min(id_x+2, vertex_idx(7, 0)); 
        }
        else
            vertex_idx(3, 0) = vertex_idx(2, 0) = vertex_idx(6, 0) = vertex_idx(7, 0) = id_x + 1;


        int z_lo = max(0, vertex_idx(4, 2) - _step_length);
        int z_up = min(_max_z_id, vertex_idx(0, 2) + _step_length);
        // Z+ now is the above side : (p1 -- p2 -- p3 -- p4) face
        // ############################################################################################################
        collide = false;
        for(id_z = vertex_idx(0, 2); id_z <= z_up; id_z++ )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(0, 1); id_y <= vertex_idx(1, 1); id_y++ ) // P1P2
            {
                if( collide == true) 
                    break;

                for(id_x = vertex_idx(0, 0); id_x >= vertex_idx(3, 0); id_x-- ) // P1P4
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(0, 2) = max(id_z-2, vertex_idx(0, 2));
            vertex_idx(1, 2) = max(id_z-2, vertex_idx(1, 2));
            vertex_idx(2, 2) = max(id_z-2, vertex_idx(2, 2));
            vertex_idx(3, 2) = max(id_z-2, vertex_idx(3, 2));
        }
        vertex_idx(0, 2) = vertex_idx(1, 2) = vertex_idx(2, 2) = vertex_idx(3, 2) = id_z - 1;

        // Z- now is the below side : (p5 -- p6 -- p7 -- p8) face
        // ############################################################################################################
        collide = false;
        for(id_z = vertex_idx(4, 2); id_z >= z_lo; id_z-- )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(4, 1); id_y <= vertex_idx(5, 1); id_y++ ) //P5P6
            {
                if( collide == true) 
                    break;

                for(id_x = vertex_idx(4, 0); id_x >= vertex_idx(7, 0); id_x-- ) // P5P8
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(4, 2) = min(id_z+2, vertex_idx(4, 2));
            vertex_idx(5, 2) = min(id_z+2, vertex_idx(5, 2));
            vertex_idx(6, 2) = min(id_z+2, vertex_idx(6, 2));
            vertex_idx(7, 2) = min(id_z+2, vertex_idx(7, 2));
        }
        else
            vertex_idx(4, 2) = vertex_idx(5, 2) = vertex_idx(6, 2) = vertex_idx(7, 2) = id_z + 1;



        if(vertex_idx_lst == vertex_idx)  // 膨胀one step,前后无变化,达到最大状态,跳出循环
            break;

        vertex_idx_lst = vertex_idx;

        MatrixXd vertex_coord(8, 3);
        for(int i = 0; i < 8; i++)
        {   
            int index_x = max(min(vertex_idx(i, 0), _max_x_id - 1), 0);  // 这里为什么是_max_x_id-1和0?
            int index_y = max(min(vertex_idx(i, 1), _max_y_id - 1), 0);
            int index_z = max(min(vertex_idx(i, 2), _max_z_id - 1), 0);

            Vector3i index(index_x, index_y, index_z);
            Vector3d pos = collision_map->GridIndexToLocation(index);
            vertex_coord.row(i) = pos;
        }

        // 使用vertex_idx继续迭代, 这里vertex_coord只是为了计算cubeMax(每次迭代后的cube)
        cubeMax.setVertex(vertex_coord, _resolution);  // 将从collision_map->GridIndexToLocation(index)获得的顶点pos放入栅格中心
        if( isContains(lstcube, cubeMax))  // 剪枝
            return make_pair(lstcube, false);

        iter ++;
    }

    return make_pair(cubeMax, true);   // 膨胀前后无变化,则达到最大状态    
}

Cube generateCube( Vector3d pt) 
{   
/*
           P4------------P3 
           /|           /|              ^
          / |          / |              | z
        P1--|---------P2 |              |
         |  P8--------|--p7             |
         | /          | /               /--------> y
         |/           |/               /  
        P5------------P6              / x
*/       
    Cube cube;
    
    pt(0) = max(min(pt(0), _pt_max_x), _pt_min_x);
    pt(1) = max(min(pt(1), _pt_max_y), _pt_min_y);
    pt(2) = max(min(pt(2), _pt_max_z), _pt_min_z);

    Vector3i pc_index = collision_map->LocationToGridIndex(pt);    
    Vector3d pc_coord = collision_map->GridIndexToLocation(pc_index);   // 注意!这里转换后的coord不是在栅格中心了

    cube.center = pc_coord;

    double x_u = pc_coord(0);
    double x_l = pc_coord(0);
    
    double y_u = pc_coord(1);
    double y_l = pc_coord(1);
    
    double z_u = pc_coord(2);
    double z_l = pc_coord(2);

    // 将cube初始化为一个点
    cube.vertex.row(0) = Vector3d(x_u, y_l, z_u);  
    cube.vertex.row(1) = Vector3d(x_u, y_u, z_u);  
    cube.vertex.row(2) = Vector3d(x_l, y_u, z_u);  
    cube.vertex.row(3) = Vector3d(x_l, y_l, z_u);  
    cube.vertex.row(4) = Vector3d(x_u, y_l, z_l);  
    cube.vertex.row(5) = Vector3d(x_u, y_u, z_l);  
    cube.vertex.row(6) = Vector3d(x_l, y_u, z_l);  
    cube.vertex.row(7) = Vector3d(x_l, y_l, z_l);  

    return cube;
}

// cube1 >= cube2
bool isContains(Cube cube1, Cube cube2) 
{   
    if( cube1.vertex(0, 0) >= cube2.vertex(0, 0) && cube1.vertex(0, 1) <= cube2.vertex(0, 1) && cube1.vertex(0, 2) >= cube2.vertex(0, 2) &&
        cube1.vertex(6, 0) <= cube2.vertex(6, 0) && cube1.vertex(6, 1) >= cube2.vertex(6, 1) && cube1.vertex(6, 2) <= cube2.vertex(6, 2)  )
        return true;
    else
        return false; 
}

void corridorSimplify(vector<Cube> & cubicList)
{
    vector<Cube> cubicSimplifyList;
    for(int j = (int)cubicList.size() - 1; j >= 0; j--)
    {   
        for(int k = j - 1; k >= 0; k--)
        {   
            if(cubicList[k].valid == false)
                continue;
            else if(isContains(cubicList[j], cubicList[k]))
                cubicList[k].valid = false;   
        }
    }

    for(auto cube:cubicList)
        if(cube.valid == true)
            cubicSimplifyList.push_back(cube);

    cubicList = cubicSimplifyList;
}

vector<Cube> corridorGeneration(vector<Vector3d> path_coord, vector<double> time)
{   
    vector<Cube> cubeList;
    Vector3d pt;

    Cube lstcube;

    for (int i = 0; i < (int)path_coord.size(); i += 1)
    {
        pt = path_coord[i];

        Cube cube = generateCube(pt);
        auto result = inflateCube(cube, lstcube);

        if(result.second == false)
            continue;

        cube = result.first;
        
        lstcube = cube;
        cube.t = time[i];
        cubeList.push_back(cube);
    }
    return cubeList;
}

vector<Cube> corridorGeneration(vector<Vector3d> path_coord)
{   
    vector<Cube> cubeList;
    Vector3d pt;

    Cube lstcube;

    for (int i = 0; i < (int)path_coord.size(); i += 1)
    {
        pt = path_coord[i];

        Cube cube = generateCube(pt);
        auto result = inflateCube(cube, lstcube);

        if(result.second == false)  // 当前路径点膨胀一次之后的cube被上一个cube完全包含,对该点进行剪枝
            continue;

        cube = result.first;
        
        lstcube = cube;
        cubeList.push_back(cube);
    }
    return cubeList;
}

double velMapping(double d, double max_v)
{   
    double vel;

    if( d <= 0.25)
        vel = 2.0 * d * d;
    else if(d > 0.25 && d <= 0.75)
        vel = 1.5 * d - 0.25;
    else if(d > 0.75 && d <= 1.0)
        vel = - 2.0 * (d - 1.0) * (d - 1.0) + 1;  
    else
        vel = 1.0;

    return vel * max_v;
}

void trajPlanning()
{   
    if( _has_target == false || _has_map == false || _has_odom == false) 
        return;

    vector<Cube> corridor;
    if(_is_use_fm)
    {
        ros::Time time_1 = ros::Time::now();
        float oob_value = INFINITY;
        auto EDT = collision_map_local->ExtractDistanceField(oob_value);
        ros::Time time_2 = ros::Time::now();
        ROS_WARN("time in generate EDT is %f", (time_2 - time_1).toSec());

        unsigned int idx;
        double max_vel = _MAX_Vel * 0.75; 
        vector<unsigned int> obs;            
        Vector3d pt;
        vector<int64_t> pt_idx;
        double flow_vel;

        unsigned int size_x = (unsigned int)(_max_x_id);
        unsigned int size_y = (unsigned int)(_max_y_id);
        unsigned int size_z = (unsigned int)(_max_z_id);

        Coord3D dimsize {size_x, size_y, size_z};
        FMGrid3D grid_fmm(dimsize);

        for(unsigned int k = 0; k < size_z; k++)
        {
            for(unsigned int j = 0; j < size_y; j++)
            {
                for(unsigned int i = 0; i < size_x; i++)
                {
                    idx = k * size_y * size_x + j * size_x + i;
                    pt << (i + 0.5) * _resolution + _map_origin(0), 
                          (j + 0.5) * _resolution + _map_origin(1), 
                          (k + 0.5) * _resolution + _map_origin(2);

                    Vector3i index = collision_map_local->LocationToGridIndex(pt);

                    if(collision_map_local->Inside(index))
                    {
                        double d = sqrt(EDT.GetImmutable(index).first.distance_square) * _resolution;
                        flow_vel = velMapping(d, max_vel);
                    }
                    else
                        flow_vel = max_vel;
    
                    if( k == 0 || k == (size_z - 1) || j == 0 || j == (size_y - 1) || i == 0 || i == (size_x - 1) )
                        flow_vel = 0.0;

                    grid_fmm[idx].setOccupancy(flow_vel);
                    if (grid_fmm[idx].isOccupied())
                        obs.push_back(idx);
                }
            }
        }
        
        grid_fmm.setOccupiedCells(std::move(obs));
        grid_fmm.setLeafSize(_resolution);

        Vector3d startIdx3d = (_start_pt - _map_origin) * _inv_resolution; 
        Vector3d endIdx3d   = (_end_pt   - _map_origin) * _inv_resolution;

        Coord3D goal_point = {(unsigned int)startIdx3d[0], (unsigned int)startIdx3d[1], (unsigned int)startIdx3d[2]};
        Coord3D init_point = {(unsigned int)endIdx3d[0],   (unsigned int)endIdx3d[1],   (unsigned int)endIdx3d[2]}; 

        unsigned int startIdx;
        vector<unsigned int> startIndices;
        grid_fmm.coord2idx(init_point, startIdx);
        
        startIndices.push_back(startIdx);
        
        unsigned int goalIdx;
        grid_fmm.coord2idx(goal_point, goalIdx);
        grid_fmm[goalIdx].setOccupancy(max_vel);     

        Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM
    
        fm_solver->setEnvironment(&grid_fmm);
        fm_solver->setInitialAndGoalPoints(startIndices, goalIdx);

        ros::Time time_bef_fm = ros::Time::now();
        if(fm_solver->compute(max_vel) == -1)
        {
            ROS_WARN("[Fast Marching Node] No path can be found");
            _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
            _traj_pub.publish(_traj);
            _has_traj = false;

            return;
        }
        ros::Time time_aft_fm = ros::Time::now();
        ROS_WARN("[Fast Marching Node] Time in Fast Marching computing is %f", (time_aft_fm - time_bef_fm).toSec() );

        Path3D path3D;
        vector<double> path_vels, time;
        GradientDescent< FMGrid3D > grad3D;
        grid_fmm.coord2idx(goal_point, goalIdx);

        if(grad3D.gradient_descent(grid_fmm, goalIdx, path3D, path_vels, time) == -1)
        {
            ROS_WARN("[Fast Marching Node] FMM failed, valid path not exists");
            if(_has_traj && _is_emerg)
            {
                _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
                _traj_pub.publish(_traj);
                _has_traj = false;
            } 
            return;
        }

        vector<Vector3d> path_coord;
        path_coord.push_back(_start_pt);

        double coord_x, coord_y, coord_z;
        for( int i = 0; i < (int)path3D.size(); i++)
        {
            coord_x = max(min( (path3D[i][0]+0.5) * _resolution + _map_origin(0), _x_size), -_x_size);
            coord_y = max(min( (path3D[i][1]+0.5) * _resolution + _map_origin(1), _y_size), -_y_size);
            coord_z = max(min( (path3D[i][2]+0.5) * _resolution, _z_size), 0.0);

            Vector3d pt(coord_x, coord_y, coord_z);
            path_coord.push_back(pt);
        }
        visPath(path_coord);

        ros::Time time_bef_corridor = ros::Time::now();    
        sortPath(path_coord, time);
        corridor = corridorGeneration(path_coord, time);
        ros::Time time_aft_corridor = ros::Time::now();
        ROS_WARN("Time consume in corridor generation is %f", (time_aft_corridor - time_bef_corridor).toSec());

        timeAllocation(corridor, time);
        visCorridor(corridor);

        delete fm_solver;
    }
    else
    {
        path_finder->linkLocalMap(collision_map_local, _local_origin);
        path_finder->AstarSearch(_start_pt, _end_pt);
        vector<Vector3d> gridPath = path_finder->getPath();
        vector<GridNodePtr> searchedNodes = path_finder->getVisitedNodes();
        path_finder->resetLocalMap();
        
        visGridPath(gridPath);
        visExpNode(searchedNodes);

        ros::Time time_bef_corridor = ros::Time::now();
        corridor = corridorGeneration(gridPath);
        ros::Time time_aft_corridor = ros::Time::now();
        ROS_WARN("Time consume in corridor generation is %f", (time_aft_corridor - time_bef_corridor).toSec());

        timeAllocation(corridor);
        visCorridor(corridor);
    }

    MatrixXd pos = MatrixXd::Zero(2,3);
    MatrixXd vel = MatrixXd::Zero(2,3);
    MatrixXd acc = MatrixXd::Zero(2,3);

    pos.row(0) = _start_pt;
    pos.row(1) = _end_pt;
    vel.row(0) = _start_vel;
    acc.row(0) = _start_acc;
    
    double obj;
    ros::Time time_bef_opt = ros::Time::now();

    if(_trajectoryGenerator.BezierPloyCoeffGeneration
        ( corridor, _MQM, pos, vel, acc, _MAX_Vel, _MAX_Acc, _traj_order, _minimize_order, 
         _cube_margin, _is_limit_vel, _is_limit_acc, obj, _bezier_coeff ) == -1 )
    {
        ROS_WARN("Cannot find a feasible and optimal solution, somthing wrong with the mosek solver");
          
        if(_has_traj && _is_emerg)  // 生成轨迹失败或发生碰撞
        {
            _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
            _traj_pub.publish(_traj);
            _has_traj = false;
        }
    }
    else
    {   
        _seg_num = corridor.size();
        _seg_time.resize(_seg_num);

        for(int i = 0; i < _seg_num; i++)
            _seg_time(i) = corridor[i].t;

        _is_emerg = false;
        _has_traj = true;

        _traj = getBezierTraj();
        _traj_pub.publish(_traj);
        _traj_id ++;  // 记录生成轨迹的次数
        visBezierTrajectory(_bezier_coeff, _seg_time);
    }

    ros::Time time_aft_opt = ros::Time::now();

    ROS_WARN("The objective of the program is %f", obj);
    ROS_WARN("The time consumation of the program is %f", (time_aft_opt - time_bef_opt).toSec());
}

void sortPath(vector<Vector3d> & path_coord, vector<double> & time)
{   
    vector<Vector3d> path_tmp;
    vector<double> time_tmp;

    for (int i = 0; i < (int)path_coord.size(); i += 1)
    {
        if( i )
            if( std::isinf(time[i]) || time[i] == 0.0 || time[i] == time[i-1] )
                continue;

        if( (path_coord[i] - _end_pt).norm() < 0.2)
            break;

        path_tmp.push_back(path_coord[i]);
        time_tmp.push_back(time[i]);
    }
    path_coord = path_tmp;
    time       = time_tmp;
}   

void timeAllocation(vector<Cube> & corridor, vector<double> time)
{   
    vector<double> tmp_time;

    for(int i  = 0; i < (int)corridor.size() - 1; i++)
    {   
        double duration  = (corridor[i].t - corridor[i+1].t);
        tmp_time.push_back(duration);
    }    
    double lst_time = corridor.back().t;
    tmp_time.push_back(lst_time);

    vector<Vector3d> points;
    points.push_back (_start_pt);
    for(int i = 1; i < (int)corridor.size(); i++)
        points.push_back(corridor[i].center);

    points.push_back (_end_pt);

    double _Vel = _MAX_Vel * 0.6;
    double _Acc = _MAX_Acc * 0.6;

    Eigen::Vector3d initv = _start_vel;
    for(int i = 0; i < (int)points.size() - 1; i++)
    {
        double dtxyz;

        Eigen::Vector3d p0   = points[i];    
        Eigen::Vector3d p1   = points[i + 1];
        Eigen::Vector3d d    = p1 - p0;            
        Eigen::Vector3d v0(0.0, 0.0, 0.0);        
        
        if( i == 0) v0 = initv;

        double D    = d.norm();                   
        double V0   = v0.dot(d / D);              
        double aV0  = fabs(V0);                   

        double acct = (_Vel - V0) / _Acc * ((_Vel > V0)?1:-1); 
        double accd = V0 * acct + (_Acc * acct * acct / 2) * ((_Vel > V0)?1:-1);
        double dcct = _Vel / _Acc;                                              
        double dccd = _Acc * dcct * dcct / 2;                                   

        if (D < aV0 * aV0 / (2 * _Acc))
        {                 
            double t1 = (V0 < 0)?2.0 * aV0 / _Acc:0.0;
            double t2 = aV0 / _Acc;
            dtxyz     = t1 + t2;                 
        }
        else if (D < accd + dccd)
        {
            double t1 = (V0 < 0)?2.0 * aV0 / _Acc:0.0;
            double t2 = (-aV0 + sqrt(aV0 * aV0 + _Acc * D - aV0 * aV0 / 2)) / _Acc;
            double t3 = (aV0 + _Acc * t2) / _Acc;
            dtxyz     = t1 + t2 + t3;    
        }
        else
        {
            double t1 = acct;                              
            double t2 = (D - accd - dccd) / _Vel;
            double t3 = dcct;
            dtxyz     = t1 + t2 + t3;
        }

        if(dtxyz < tmp_time[i] * 0.5)
            tmp_time[i] = dtxyz; // if FM given time in this segment is rediculous long, use the new value
    }

    for(int i = 0; i < (int)corridor.size(); i++)
        corridor[i].t = tmp_time[i];
}

// 满足梯形速度曲线, 这个函数实现的好像有问题
void timeAllocation(vector<Cube> & corridor)
{   
    vector<Vector3d> points;
    points.push_back (_start_pt);

    // 计算出的corridor的特点:第一个cube的边界点为第二个cube的center
    for(int i = 1; i < (int)corridor.size(); i++)
        points.push_back(corridor[i].center);

    points.push_back (_end_pt);

    double _Vel = _MAX_Vel * 0.6;
    double _Acc = _MAX_Acc * 0.6;

    for (int k = 0; k < (int)points.size() - 1; k++)
    {
        double dtxyz;
        Vector3d p0 = points[k];
        Vector3d p1 = points[k + 1];
        Vector3d d = p1 - p0;
        Vector3d v0(0.0, 0.0, 0.0);

        if (k == 0)
            v0 = _start_vel;    // _start_vel从odom中获得,为起始点的速度

        double D = d.norm();    // 相邻两点的距离
        double V0 = v0.dot(d / D);  // V0的含义???  V0 > 0:速度方向与目标点方向相同, V0 < 0:速度方向与目标点方向相反
        double aV0 = fabs(V0);

        double acct = (_Vel - V0) / _Acc * ((_Vel > V0) ? 1 : -1);  // 加速时间
        double accd = V0 * acct + (_Acc * acct * acct / 2) * ((_Vel > V0) ? 1 : -1);  // 加速位移
        double dcct = _Vel / _Acc;  // 减速时间
        double dccd = _Acc * dcct * dcct / 2;  // 减速位移

        if (D < aV0 * aV0 / (2 * _Acc))    // 两点之间距离小于加速距离, 这行写错了吧???, 测试结果:一直不执行
        {
            double t1 = (V0 < 0) ? 2.0 * aV0 / _Acc : 0.0;
            double t2 = aV0 / _Acc;
            dtxyz = t1 + t2;
        }
        else if (D < accd + dccd)    // 两点之间距离小于加速距离+减速距离
        {
            double t1 = (V0 < 0) ? 2.0 * aV0 / _Acc : 0.0;
            double t2 = (-aV0 + sqrt(aV0 * aV0 + _Acc * D - aV0 * aV0 / 2)) / _Acc;
            double t3 = (aV0 + _Acc * t2) / _Acc;
            dtxyz = t1 + t2 + t3;  
        }
        else    // 正常情况,两点之间距离=加速距离+匀速距离+减速距离
        {
            double t1 = acct;
            double t2 = (D - accd - dccd) / _Vel;
            double t3 = dcct;
            dtxyz = t1 + t2 + t3;
        }
        corridor[k].t = dtxyz;  // 一共points.size()-1条轨迹
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "b_traj_node");
    ros::NodeHandle nh("~");

    _map_sub  = nh.subscribe( "map",       1, rcvPointCloudCallBack );  // 订阅局部地图信息, 填充障碍和检查轨迹
    _odom_sub = nh.subscribe( "odometry",  1, rcvOdometryCallbck);      // 订阅里程计信息, 更新当前位置和发布无人机的tf
    _pts_sub  = nh.subscribe( "waypoints", 1, rcvWaypointsCallback );   // 订阅目标点信息, 生成第一条轨迹

    _inf_map_vis_pub   = nh.advertise<sensor_msgs::PointCloud2>("vis_map_inflate", 1);
    _local_map_vis_pub = nh.advertise<sensor_msgs::PointCloud2>("vis_map_local", 1);
    _traj_vis_pub      = nh.advertise<visualization_msgs::Marker>("trajectory_vis", 1);    
    _corridor_vis_pub  = nh.advertise<visualization_msgs::MarkerArray>("corridor_vis", 1);
    _fm_path_vis_pub   = nh.advertise<visualization_msgs::MarkerArray>("path_vis", 1);
    _grid_path_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("grid_path_vis", 1);
    _nodes_vis_pub     = nh.advertise<visualization_msgs::Marker>("expanded_nodes_vis", 1);
    _checkTraj_vis_pub = nh.advertise<visualization_msgs::Marker>("check_trajectory", 1);
    _stopTraj_vis_pub  = nh.advertise<visualization_msgs::Marker>("stop_trajectory", 1);

    _traj_pub = nh.advertise<quadrotor_msgs::PolynomialTrajectory>("trajectory", 10);

    nh.param("map/margin",     _cloud_margin, 0.25);    // 障碍物膨胀半径
    nh.param("map/resolution", _resolution, 0.2);
    
    nh.param("map/x_size",       _x_size, 50.0);
    nh.param("map/y_size",       _y_size, 50.0);
    nh.param("map/z_size",       _z_size, 5.0 );
    
    nh.param("map/x_local_size", _x_local_size, 20.0);
    nh.param("map/y_local_size", _y_local_size, 20.0);
    nh.param("map/z_local_size", _z_local_size, 5.0 );

    nh.param("planning/init_x",       _init_x,  0.0);
    nh.param("planning/init_y",       _init_y,  0.0);
    nh.param("planning/init_z",       _init_z,  0.0);
    
    nh.param("planning/max_vel",       _MAX_Vel,  1.0);
    nh.param("planning/max_acc",       _MAX_Acc,  1.0);
    nh.param("planning/max_inflate",   _max_inflate_iter, 100);  // 膨胀cube时的最大迭代次数
    nh.param("planning/step_length",   _step_length,     2);     // 膨胀cube时的步长
    nh.param("planning/cube_margin",   _cube_margin,   0.2);
    nh.param("planning/check_horizon", _check_horizon,10.0);     // 检查碰撞的范围, 单位: s
    nh.param("planning/stop_horizon",  _stop_horizon,  5.0);     // 终止运动的碰撞范围, 单位: s
    nh.param("planning/is_limit_vel",  _is_limit_vel,  false);
    nh.param("planning/is_limit_acc",  _is_limit_acc,  false);
    nh.param("planning/is_use_fm",     _is_use_fm,  true);

    nh.param("optimization/min_order",  _minimize_order, 3.0);  // 优化过程中, minimize的阶数  1 -> velocity, 2 -> acceleration, 3 -> jerk, 4 -> snap
    nh.param("optimization/poly_order", _traj_order,     10);   // 轨迹的阶数, 系数的个数为_traj_order+1

    nh.param("vis/vis_traj_width", _vis_traj_width, 0.15);
    nh.param("vis/is_proj_cube",   _is_proj_cube, true);

    Bernstein _bernstein;
    if(_bernstein.setParam(3, 12, _minimize_order) == -1)
        ROS_ERROR(" The trajectory order is set beyond the library's scope, please re-set "); 

    // 获得所选多项式阶数对应的参数, 注意这只是one block
    _MQM = _bernstein.getMQM()[_traj_order];
    _FM  = _bernstein.getFM()[_traj_order];
    _C   = _bernstein.getC()[_traj_order];
    _Cv  = _bernstein.getC_v()[_traj_order];
    _Ca  = _bernstein.getC_a()[_traj_order];
    _Cj  = _bernstein.getC_j()[_traj_order];

    _map_origin << -_x_size/2.0, -_y_size/2.0, 0.0;    // -25,-25,0
    _pt_max_x = + _x_size / 2.0;
    _pt_min_x = - _x_size / 2.0;
    _pt_max_y = + _y_size / 2.0;
    _pt_min_y = - _y_size / 2.0; 
    _pt_max_z = + _z_size;
    _pt_min_z = 0.0;

    _inv_resolution = 1.0 / _resolution;
    _max_x_id = (int)(_x_size * _inv_resolution);
    _max_y_id = (int)(_y_size * _inv_resolution);
    _max_z_id = (int)(_z_size * _inv_resolution);
    _max_local_x_id = (int)(_x_local_size * _inv_resolution);
    _max_local_y_id = (int)(_y_local_size * _inv_resolution);
    _max_local_z_id = (int)(_z_local_size * _inv_resolution);

    Vector3i GLSIZE(_max_x_id, _max_y_id, _max_z_id);     // global_map size
    Vector3i LOSIZE(_max_local_x_id, _max_local_y_id, _max_local_z_id);     // local_map size

    path_finder = new gridPathFinder(GLSIZE, LOSIZE);
    path_finder->initGridNodeMap(_resolution, _map_origin);  // 初始化全局网格地图

    Translation3d origin_translation( _map_origin(0), _map_origin(1), 0.0);
    Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);    // 0°
    Affine3d origin_transform = origin_translation * origin_rotation;
    // global_map
    collision_map = new CollisionMapGrid(origin_transform, "world", _resolution, _x_size, _y_size, _z_size, _free_cell);

    ros::Rate rate(100);
    bool status = ros::ok();
    while(status) 
    {
        ros::spinOnce();           
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}

// 将一条完整的轨迹信息封装到msg中
quadrotor_msgs::PolynomialTrajectory getBezierTraj()
{
    quadrotor_msgs::PolynomialTrajectory traj;
      traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;
      traj.num_segment = _seg_num;

      int order = _traj_order;
      int poly_num1d = order + 1;
      int polyTotalNum = _seg_num * (order + 1);

      traj.coef_x.resize(polyTotalNum);
      traj.coef_y.resize(polyTotalNum);
      traj.coef_z.resize(polyTotalNum);

      int idx = 0;
      for(int i = 0; i < _seg_num; i++ )
      {    
          for(int j =0; j < poly_num1d; j++)
          { 
              traj.coef_x[idx] = _bezier_coeff(i,                  j);  // x
              traj.coef_y[idx] = _bezier_coeff(i,     poly_num1d + j);  // y
              traj.coef_z[idx] = _bezier_coeff(i, 2 * poly_num1d + j);  // z
              idx++;
          }
      }

      traj.header.frame_id = "/bernstein";
      traj.header.stamp = _odom.header.stamp;  // 注意, 以里程计的时间戳作为轨迹是时间戳
      _start_time = traj.header.stamp;  // 检查轨迹是否碰撞时会使用

      traj.time.resize(_seg_num);
      traj.order.resize(_seg_num);

      traj.mag_coeff = 1.0;
      for (int idx = 0; idx < _seg_num; ++idx)
      {
          traj.time[idx] = _seg_time(idx);
          traj.order[idx] = _traj_order;
      }

      traj.start_yaw = 0.0;
      traj.final_yaw = 0.0;

      traj.trajectory_id = _traj_id;

      return traj;
}

Vector3d getPosFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now )
{
    Vector3d ret = VectorXd::Zero(3); // x, y, z轴的pos
    VectorXd ctrl_now = polyCoeff.row(seg_now);
    int ctrl_num1D = polyCoeff.cols() / 3;

    for (int i = 0; i < 3; i++) // x, y, z
        for (int j = 0; j < ctrl_num1D; j++)
            // ret(i) += Cnj * Cj * t^j * (1-t)^(n-j), 贝塞尔曲线的标准形式
            ret(i) += _C(j) * ctrl_now(i * ctrl_num1D + j) * pow(t_now, j) * pow((1 - t_now), (_traj_order - j));

    return ret;
}

VectorXd getStateFromBezier(const MatrixXd & polyCoeff, double t_now, int seg_now )
{
    VectorXd ret = VectorXd::Zero(12);

    VectorXd ctrl_now = polyCoeff.row(seg_now);
    int ctrl_num1D = polyCoeff.cols() / 3;

    for(int i = 0; i < 3; i++)
    {   
        for(int j = 0; j < ctrl_num1D; j++){
            ret[i] += _C(j) * ctrl_now(i * ctrl_num1D + j) * pow(t_now, j) * pow((1 - t_now), (_traj_order - j) ); 
          
            if(j < ctrl_num1D - 1 )
                ret[i+3] += _Cv(j) * _traj_order 
                      * ( ctrl_now(i * ctrl_num1D + j + 1) - ctrl_now(i * ctrl_num1D + j))
                      * pow(t_now, j) * pow((1 - t_now), (_traj_order - j - 1) ); 
          
            if(j < ctrl_num1D - 2 )
                ret[i+6] += _Ca(j) * _traj_order * (_traj_order - 1) 
                      * ( ctrl_now(i * ctrl_num1D + j + 2) - 2 * ctrl_now(i * ctrl_num1D + j + 1) + ctrl_now(i * ctrl_num1D + j))
                      * pow(t_now, j) * pow((1 - t_now), (_traj_order - j - 2) );                         

            if(j < ctrl_num1D - 3 )
                ret[i+9] += _Cj(j) * _traj_order * (_traj_order - 1) * (_traj_order - 2) 
                      * ( ctrl_now(i * ctrl_num1D + j + 3) - 3 * ctrl_now(i * ctrl_num1D + j + 2) + 3 * ctrl_now(i * ctrl_num1D + j + 1) - ctrl_now(i * ctrl_num1D + j))
                      * pow(t_now, j) * pow((1 - t_now), (_traj_order - j - 3) );                         
        }
    }

    return ret;  
}

visualization_msgs::MarkerArray path_vis; 
void visPath(vector<Vector3d> path)
{
    for(auto & mk: path_vis.markers) 
        mk.action = visualization_msgs::Marker::DELETE;

    _fm_path_vis_pub.publish(path_vis);
    path_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.ns = "b_traj/fast_marching_path";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.a = 0.6;
    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;

    int idx = 0;
    for(int i = 0; i < int(path.size()); i++)
    {
        mk.id = idx;

        mk.pose.position.x = path[i](0); 
        mk.pose.position.y = path[i](1); 
        mk.pose.position.z = path[i](2);  

        mk.scale.x = _resolution;
        mk.scale.y = _resolution;
        mk.scale.z = _resolution;

        idx ++;
        path_vis.markers.push_back(mk);
    }

    _fm_path_vis_pub.publish(path_vis);
}

visualization_msgs::MarkerArray cube_vis;
void visCorridor(vector<Cube> corridor)
{   
    for(auto & mk: cube_vis.markers) 
        mk.action = visualization_msgs::Marker::DELETE;  // 删除上一次的cube
    
    _corridor_vis_pub.publish(cube_vis);

    cube_vis.markers.clear();  // 和DELETE操作重复

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.ns = "corridor";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;

    mk.color.a = 0.4;
    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;

    int idx = 0;
    for(int i = 0; i < int(corridor.size()); i++)
    {   
        mk.id = idx;

        mk.pose.position.x = (corridor[i].vertex(0, 0) + corridor[i].vertex(3, 0) ) / 2.0; 
        mk.pose.position.y = (corridor[i].vertex(0, 1) + corridor[i].vertex(1, 1) ) / 2.0; 

        if(_is_proj_cube)
            mk.pose.position.z = 0.0;   // 二维
        else
            mk.pose.position.z = (corridor[i].vertex(0, 2) + corridor[i].vertex(4, 2) ) / 2.0; 

        mk.scale.x = (corridor[i].vertex(0, 0) - corridor[i].vertex(3, 0) );
        mk.scale.y = (corridor[i].vertex(1, 1) - corridor[i].vertex(0, 1) );

        if(_is_proj_cube)
            mk.scale.z = 0.05; 
        else
            mk.scale.z = (corridor[i].vertex(0, 2) - corridor[i].vertex(4, 2) );

        idx ++;
        cube_vis.markers.push_back(mk);
    }

    _corridor_vis_pub.publish(cube_vis);
}

void visBezierTrajectory(MatrixXd polyCoeff, VectorXd time)
{   
    visualization_msgs::Marker traj_vis;

    traj_vis.header.stamp       = ros::Time::now();
    traj_vis.header.frame_id    = "world";

    traj_vis.ns = "trajectory/trajectory";
    traj_vis.id = 0;
    traj_vis.type = visualization_msgs::Marker::SPHERE_LIST;
    
    traj_vis.action = visualization_msgs::Marker::DELETE;
    _checkTraj_vis_pub.publish(traj_vis); // 删除以前的轨迹
    _stopTraj_vis_pub.publish(traj_vis);

    traj_vis.action = visualization_msgs::Marker::ADD;
    traj_vis.scale.x = _vis_traj_width;
    traj_vis.scale.y = _vis_traj_width;
    traj_vis.scale.z = _vis_traj_width;
    traj_vis.pose.orientation.x = 0.0;
    traj_vis.pose.orientation.y = 0.0;
    traj_vis.pose.orientation.z = 0.0;
    traj_vis.pose.orientation.w = 1.0;
    traj_vis.color.r = 1.0;
    traj_vis.color.g = 0.0;
    traj_vis.color.b = 0.0;
    traj_vis.color.a = 0.6;

    double traj_len = 0.0;
    int count = 0;
    Vector3d cur, pre;
    cur.setZero();
    pre.setZero();
    
    traj_vis.points.clear(); // 清空以前的轨迹点

    Vector3d state;
    geometry_msgs::Point pt;

    int segment_num = polyCoeff.rows();
    for (int i = 0; i < segment_num; i++)
    {
        // 因为是标准的贝塞尔曲线, 所以需考虑缩放系数
        for (double t = 0.0; t < 1.0; t += 0.05 / time(i), count++)
        {
            state = getPosFromBezier(polyCoeff, t, i); // 从标准的贝塞尔曲线得到pos
            cur(0) = pt.x = time(i) * state(0); // 这里的time(i)是比例系数,将坐标放大到真实值
            cur(1) = pt.y = time(i) * state(1);
            cur(2) = pt.z = time(i) * state(2);
            traj_vis.points.push_back(pt);

            if (count)
                traj_len += (pre - cur).norm(); // 轨迹长度
            pre = cur;
        }
    }

    ROS_INFO("[GENERATOR] The length of the trajectory; %.3lfm.", traj_len);
    _traj_vis_pub.publish(traj_vis);
}

visualization_msgs::MarkerArray grid_vis; 
void visGridPath( vector<Vector3d> grid_path )
{   
    for(auto & mk: grid_vis.markers) 
        mk.action = visualization_msgs::Marker::DELETE;   // 删除上次搜索的路径

    _grid_path_vis_pub.publish(grid_vis);   // 更新
    grid_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.ns = "b_traj/grid_path";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.a = 1.0;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;

    int idx = 0;
    for(int i = 0; i < int(grid_path.size()); i++)
    {
        mk.id = idx;

        mk.pose.position.x = grid_path[i](0); 
        mk.pose.position.y = grid_path[i](1); 
        mk.pose.position.z = grid_path[i](2);  

        mk.scale.x = _resolution;
        mk.scale.y = _resolution;
        mk.scale.z = _resolution;

        idx ++;
        grid_vis.markers.push_back(mk);
    }

    _grid_path_vis_pub.publish(grid_vis);
}

void visExpNode( vector<GridNodePtr> nodes )
{   
    visualization_msgs::Marker node_vis; 
    node_vis.header.frame_id = "world";
    node_vis.header.stamp = ros::Time::now();
    node_vis.ns = "b_traj/visited_nodes";
    node_vis.type = visualization_msgs::Marker::CUBE_LIST;
    node_vis.action = visualization_msgs::Marker::ADD;
    node_vis.id = 0;

    node_vis.pose.orientation.x = 0.0;
    node_vis.pose.orientation.y = 0.0;
    node_vis.pose.orientation.z = 0.0;
    node_vis.pose.orientation.w = 1.0;
    node_vis.color.a = 0.3;
    node_vis.color.r = 0.0;
    node_vis.color.g = 1.0;
    node_vis.color.b = 0.0;

    node_vis.scale.x = _resolution;
    node_vis.scale.y = _resolution;
    node_vis.scale.z = _resolution;

    geometry_msgs::Point pt;
    for(int i = 0; i < int(nodes.size()); i++)
    {
        Vector3d coord = nodes[i]->coord;
        pt.x = coord(0);
        pt.y = coord(1);
        pt.z = coord(2);

        node_vis.points.push_back(pt);
    }

    _nodes_vis_pub.publish(node_vis);
}
