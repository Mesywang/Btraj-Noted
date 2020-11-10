#include "trajectory_generator.h"
using namespace std;    
using namespace Eigen;

static void MSKAPI printstr(void *handle, MSKCONST char str[])
{
  printf("%s",str);
}

// https://docs.mosek.com/8.1/capi/tutorial-qo-shared.html
int TrajectoryGenerator::BezierPloyCoeffGeneration(
            const vector<Cube> &corridor,
            const MatrixXd &MQM,
            const MatrixXd &pos,
            const MatrixXd &vel,
            const MatrixXd &acc,
            const double maxVel,
            const double maxAcc,
            const int traj_order,
            const double minimize_order,
            const double margin,
            const bool & isLimitVel,
            const bool & isLimitAcc,
            double & obj,
            MatrixXd & PolyCoeff)  // define the order to which we minimize.   1 -- velocity, 2 -- acceleration, 3 -- jerk, 4 -- snap  
{   
#define ENFORCE_VEL  isLimitVel // whether or not adding extra constraints for ensuring the velocity feasibility
#define ENFORCE_ACC  isLimitAcc // whether or not adding extra constraints for ensuring the acceleration feasibility

    double initScale = corridor.front().t;
    double lstScale  = corridor.back().t;
    int segment_num  = corridor.size();

    int n_poly = traj_order + 1;
    int s1d1CtrlP_num = n_poly; // 控制点数量:一段轨迹, 一维坐标
    int s1CtrlP_num   = 3 * s1d1CtrlP_num; // 控制点数量:一段轨迹, 三维坐标

    int equ_con_s_num = 3 * 3; // 起点p v a, 三轴
    int equ_con_e_num = 3 * 3; // 终点p v a, 三轴
    int equ_con_continuity_num = 3 * 3 * (segment_num - 1); // 中间点 p v a, 三轴
    int equ_con_num   = equ_con_s_num + equ_con_e_num + equ_con_continuity_num; // 所有轨迹的端点 p v a, 三轴
    
    int vel_con_num = 3 *  traj_order * segment_num; // v的所有控制点, 三轴
    int acc_con_num = 3 * (traj_order - 1) * segment_num; // a的所有控制点, 三轴

    if( !ENFORCE_VEL )
        vel_con_num = 0;

    if( !ENFORCE_ACC )
        acc_con_num = 0;

    int high_order_con_num = vel_con_num + acc_con_num; 
    //int high_order_con_num = 0; //3 * traj_order * segment_num;

    int con_num   = equ_con_num + high_order_con_num;
    int ctrlP_num = segment_num * s1CtrlP_num; // 控制点数量:整条轨迹, 三维坐标

    double x_var[ctrlP_num]; // 优化的结果(所有控制点)
    double primalobj;

    MSKrescodee  r; 

    /*******************************构造约束条件的界限(包括等式约束和不等式约束)*******************************/
    vector< pair<MSKboundkeye, pair<double, double> > > con_bdk; 
    
    if(ENFORCE_VEL)
    {
        // 设定速度的上下限(不等式约束)
        for(int i = 0; i < vel_con_num; i++)
        {
            // cb_ie: 不等式约束条件的上下限(所有点的速度)
            pair<MSKboundkeye, pair<double, double> > cb_ie = make_pair( MSK_BK_RA, make_pair( - maxVel,  + maxVel) );
            con_bdk.push_back(cb_ie);   
        }
    }
    // 关于MSK_BK_RA, MSK_BK_FX, MSK_BK_LO等的含义: https://docs.mosek.com/8.1/capi/conventions.html#doc-optimizer-cmo-rmo-matrix
    if(ENFORCE_ACC)
    {
        // 设定加速度的上下限(不等式约束)
        for(int i = 0; i < acc_con_num; i++)
        {
            // cb_ie: 不等式约束条件的上下限(所有点的加速度)
            pair<MSKboundkeye, pair<double, double> > cb_ie = make_pair( MSK_BK_RA, make_pair( - maxAcc,  maxAcc) ); 
            con_bdk.push_back(cb_ie);   
        }
    }

    //ROS_WARN("[Bezier Trajectory] equality bound %d", equ_con_num);
    for (int i = 0; i < equ_con_num; i++)
    {
        double beq_i;
        if (i < 3)
            beq_i = pos(0, i);        // 起点 pos x, y, z
        else if (i >= 3 && i < 6)
            beq_i = vel(0, i - 3);    // 起点 vel x, y, z
        else if (i >= 6 && i < 9)
            beq_i = acc(0, i - 6);    // 起点 acc x, y, z 
        else if (i >= 9 && i < 12)
            beq_i = pos(1, i - 9);    // 终点 pos x, y, z
        else if (i >= 12 && i < 15)
            beq_i = vel(1, i - 12);   // 终点 vel x, y, z
        else if (i >= 15 && i < 18)
            beq_i = acc(1, i - 15);   // 终点 acc x, y, z   
        else
            beq_i = 0.0;              // 中间点连续x, y, z

        // cb_eq: 等式约束条件的上下限,包括:起点p v a, 终点p v a, 中间点连续. 在Mosek中等式约束用上下限相同来表示.
        pair<MSKboundkeye, pair<double, double>> cb_eq = make_pair(MSK_BK_FX, make_pair(beq_i, beq_i)); 
        con_bdk.push_back(cb_eq);
    }

    /*******************************构造优化变量x的界限(贝塞尔曲线的控制点)*******************************/
    // 为控制点的boundkey和boundary定义一个容器
    vector< pair<MSKboundkeye, pair<double, double> > > var_bdk; 

    for(int k = 0; k < segment_num; k++)
    {   
        Cube cube_     = corridor[k];
        double scale_k = cube_.t;

        for(int i = 0; i < 3; i++ ) // x, y, z
        {   
            for(int j = 0; j < n_poly; j ++ ) // 所有控制点
            {   
                pair<MSKboundkeye, pair<double, double> > vb_x;

                // 这里将所有控制点都约束在对应段轨迹的box中,而没有单独地将两段轨迹的连接点约束在相邻两个box的公共区域内原因如下:
                // 前一段轨迹末控制点在前一个box的范围内,后一段轨迹末控制点在后一个box的范围内,再加上两个控制点相等这一条件,那么
                // 这个控制点就自动地被限定在相邻两个box的公共区域内了,不需要再另外设置约束了.(妙啊)
                double lo_bound, up_bound;
                if(k > 0)
                {
                    lo_bound = (cube_.box[i].first  + margin) / scale_k; // scale_k为缩放系数
                    up_bound = (cube_.box[i].second - margin) / scale_k;
                }
                else
                {
                    lo_bound = (cube_.box[i].first)  / scale_k;
                    up_bound = (cube_.box[i].second) / scale_k;
                }

                // vb_x: 待优化变量的界限(此问题指分段多项式的系数,也就是贝塞尔曲线的控制点)
                vb_x  = make_pair( MSK_BK_RA, make_pair( lo_bound, up_bound ) ); 

                var_bdk.push_back(vb_x);
            }
        } 
    }

    /*******************************构造Mosek环境*******************************/
    MSKint32t  j,i; 
    MSKenv_t   env; 
    MSKtask_t  task; 
    // Create the mosek environment. 
    r = MSK_makeenv( &env, NULL ); 
  
    // Create the optimization task.
    r = MSK_maketask(env, con_num, ctrlP_num, &task);   // con_num, ctrlP_num分别为约束和优化变量的个数

    // Parameters used in the optimizer
    //######################################################################
    //MSK_putintparam (task, MSK_IPAR_OPTIMIZER , MSK_OPTIMIZER_INTPNT );
    MSK_putintparam (task, MSK_IPAR_NUM_THREADS, 1);
    MSK_putdouparam (task, MSK_DPAR_CHECK_CONVEXITY_REL_TOL, 1e-2);
    MSK_putdouparam (task, MSK_DPAR_INTPNT_TOL_DFEAS,  1e-4);
    MSK_putdouparam (task, MSK_DPAR_INTPNT_TOL_PFEAS,  1e-4);
    MSK_putdouparam (task, MSK_DPAR_INTPNT_TOL_INFEAS, 1e-4);
    //MSK_putdouparam (task, MSK_DPAR_INTPNT_TOL_REL_GAP, 5e-2 );
    //######################################################################

    //r = MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr);

    // Append empty constraints.
    //The constraints will initially have no bounds.
    if ( r == MSK_RES_OK ) 
      r = MSK_appendcons(task,con_num);  

    // Append optimizing variables. The variables will initially be fixed at zero (x=0). 
    if ( r == MSK_RES_OK ) 
      r = MSK_appendvars(task,ctrlP_num); 

    // Set the bounds on variable j.
    // blx[j] <= x_j <= bux[j]
    for ( j = 0; j < ctrlP_num && r == MSK_RES_OK; ++j )
    {
        if (r == MSK_RES_OK)
            r = MSK_putvarbound(task,
                                j,                         // Index of variable.
                                var_bdk[j].first,          // Bound key.
                                var_bdk[j].second.first,   // Numerical value of lower bound.
                                var_bdk[j].second.second); // Numerical value of upper bound.
    }

    // Set the bounds on constraints. 
    // for i=1, ...,con_num : blc[i] <= constraint i <= buc[i]
    for ( i = 0; i < con_num && r == MSK_RES_OK; i++ )
    {
        if (r == MSK_RES_OK)
            r = MSK_putconbound(task,
                                i,                         // Index of constraint.
                                con_bdk[i].first,          // Bound key.
                                con_bdk[i].second.first,   // Numerical value of lower bound.
                                con_bdk[i].second.second); // Numerical value of upper bound.
    }


    /*******************************构造等式约束和不等式约束对应的线性矩阵A*******************************/
    //ROS_WARN("[Bezier Trajectory] Start stacking the Linear Matrix A, inequality part");
    int row_idx = 0;   // A矩阵的行索引
    
    // The velocity constraints
    if(ENFORCE_VEL)
    {   
        for(int k = 0; k < segment_num ; k ++ )
        {   
            for(int i = 0; i < 3; i++)
            {  // for x, y, z loop
                for(int p = 0; p < traj_order; p++)
                {
                    int nzi = 2;  // 每一行中非0元素的个数
                    MSKint32t asub[nzi];
                    double aval[nzi];

                    aval[0] = -1.0 * traj_order;
                    aval[1] =  1.0 * traj_order;

                    // 行: row_idx, 列: asub[]
                    // 顺序如下: 第一段轨迹x轴c0~cn, 第一段轨迹y轴c0~cn, 第一段轨迹z轴c0~cn, 第二段轨迹x轴c0~cn .....
                    asub[0] = k * s1CtrlP_num + i * s1d1CtrlP_num + p;    
                    asub[1] = k * s1CtrlP_num + i * s1d1CtrlP_num + p + 1;    

                    // 关于MSK_putarow的用法: https://docs.mosek.com/8.1/capi/alphabetic-functionalities.html#mosek.task.putarow
                    r = MSK_putarow(task, row_idx, nzi, asub, aval);    
                    row_idx ++;
                }
            }
        }
    }

    // The acceleration constraints
    if(ENFORCE_ACC)
    {
        for(int k = 0; k < segment_num ; k ++ )
        {
            for(int i = 0; i < 3; i++)
            { 
                for(int p = 0; p < traj_order - 1; p++)
                {    
                    int nzi = 3;
                    MSKint32t asub[nzi];
                    double aval[nzi];

                    // corridor[k].t为比例因子, 结论可参考论文
                    aval[0] =  1.0 * traj_order * (traj_order - 1) / corridor[k].t;
                    aval[1] = -2.0 * traj_order * (traj_order - 1) / corridor[k].t;
                    aval[2] =  1.0 * traj_order * (traj_order - 1) / corridor[k].t;
                    asub[0] = k * s1CtrlP_num + i * s1d1CtrlP_num + p;    
                    asub[1] = k * s1CtrlP_num + i * s1d1CtrlP_num + p + 1;    
                    asub[2] = k * s1CtrlP_num + i * s1d1CtrlP_num + p + 2;    
                    
                    r = MSK_putarow(task, row_idx, nzi, asub, aval);    
                    row_idx ++;
                }
            }
        }
    }
    /*   Start position  */
    {
        // position :
        for(int i = 0; i < 3; i++)  // loop for x, y, z  
        {      
            int nzi = 1;
            MSKint32t asub[nzi];
            double aval[nzi];
            aval[0] = 1.0 * initScale;
            asub[0] = i * s1d1CtrlP_num;
            r = MSK_putarow(task, row_idx, nzi, asub, aval);    
            row_idx ++;
        }
        // velocity :
        for(int i = 0; i < 3; i++)  // loop for x, y, z  
        {     
            int nzi = 2;
            MSKint32t asub[nzi];
            double aval[nzi];
            aval[0] = - 1.0 * traj_order;
            aval[1] =   1.0 * traj_order;
            asub[0] = i * s1d1CtrlP_num;
            asub[1] = i * s1d1CtrlP_num + 1;
            r = MSK_putarow(task, row_idx, nzi, asub, aval);   
            row_idx ++;
        }
        // acceleration : 
        for(int i = 0; i < 3; i++)  // loop for x, y, z 
        {      
            int nzi = 3;
            MSKint32t asub[nzi];
            double aval[nzi];
            aval[0] =   1.0 * traj_order * (traj_order - 1) / initScale;
            aval[1] = - 2.0 * traj_order * (traj_order - 1) / initScale;
            aval[2] =   1.0 * traj_order * (traj_order - 1) / initScale;
            asub[0] = i * s1d1CtrlP_num;
            asub[1] = i * s1d1CtrlP_num + 1;
            asub[2] = i * s1d1CtrlP_num + 2;
            r = MSK_putarow(task, row_idx, nzi, asub, aval);    
            row_idx ++;
        }
    }      

    /*   End position  */
    {   
        // position :
        for(int i = 0; i < 3; i++)
        {  // loop for x, y, z       
            int nzi = 1;
            MSKint32t asub[nzi];
            double aval[nzi];
            asub[0] = ctrlP_num - 1 - (2 - i) * s1d1CtrlP_num;
            aval[0] = 1.0 * lstScale;
            r = MSK_putarow(task, row_idx, nzi, asub, aval);    
            row_idx ++;
        }
        // velocity :
        for(int i = 0; i < 3; i++)
        { 
            int nzi = 2;
            MSKint32t asub[nzi];
            double aval[nzi];
            asub[0] = ctrlP_num - 1 - (2 - i) * s1d1CtrlP_num - 1;
            asub[1] = ctrlP_num - 1 - (2 - i) * s1d1CtrlP_num;
            aval[0] = - 1.0;
            aval[1] =   1.0;
            r = MSK_putarow(task, row_idx, nzi, asub, aval);    
            row_idx ++;
        }
        // acceleration : 
        for(int i = 0; i < 3; i++)
        { 
            int nzi = 3;
            MSKint32t asub[nzi];
            double aval[nzi];
            asub[0] = ctrlP_num - 1 - (2 - i) * s1d1CtrlP_num - 2;
            asub[1] = ctrlP_num - 1 - (2 - i) * s1d1CtrlP_num - 1;
            asub[2] = ctrlP_num - 1 - (2 - i) * s1d1CtrlP_num;
            aval[0] =   1.0 / lstScale;
            aval[1] = - 2.0 / lstScale;
            aval[2] =   1.0 / lstScale;
            r = MSK_putarow(task, row_idx, nzi, asub, aval);    
            row_idx ++;
        }
    }

    /*   joint points  */
    {
        int sub_shift = 0;
        double val0, val1;
        for(int k = 0; k < (segment_num - 1); k ++ )
        {   
            // 前后两段轨迹的缩放比例不一样
            double scale_k = corridor[k].t;
            double scale_n = corridor[k+1].t;
            // position :
            val0 = scale_k;
            val1 = scale_n;
            for(int i = 0; i < 3; i++)  // loop for x, y, z
            {
                int nzi = 2;
                MSKint32t asub[nzi];
                double aval[nzi];

                // This segment's last control point
                aval[0] = 1.0 * val0;
                asub[0] = sub_shift + (i+1) * s1d1CtrlP_num - 1;

                // Next segment's first control point
                aval[1] = -1.0 * val1;
                asub[1] = sub_shift + s1CtrlP_num + i * s1d1CtrlP_num;  // 相邻两段轨迹的x轴中间隔着前一段轨迹的y,z轴
                r = MSK_putarow(task, row_idx, nzi, asub, aval);    
                row_idx ++;
            }
            
            for(int i = 0; i < 3; i++)  // loop for x, y, z
            {  
                int nzi = 4;
                MSKint32t asub[nzi];
                double aval[nzi];
                
                // This segment's last velocity control point
                aval[0] = -1.0;  // 这里省略了n(n-1),因为等式右边为0,可以约掉
                aval[1] =  1.0;
                asub[0] = sub_shift + (i+1) * s1d1CtrlP_num - 2;    
                asub[1] = sub_shift + (i+1) * s1d1CtrlP_num - 1;   
                // Next segment's first velocity control point
                aval[2] =  1.0;
                aval[3] = -1.0;

                asub[2] = sub_shift + s1CtrlP_num + i * s1d1CtrlP_num;    
                asub[3] = sub_shift + s1CtrlP_num + i * s1d1CtrlP_num + 1;

                r = MSK_putarow(task, row_idx, nzi, asub, aval);    
                row_idx ++;
            }
            // acceleration :
            val0 = 1.0 / scale_k;
            val1 = 1.0 / scale_n;
            for(int i = 0; i < 3; i++)  // loop for x, y, z
            {  
                int nzi = 6;
                MSKint32t asub[nzi];
                double aval[nzi];
                
                // This segment's last velocity control point
                aval[0] =  1.0  * val0;  // 注意,此时val0 = 1.0 / scale_k, val1 = 1.0 / scale_n
                aval[1] = -2.0  * val0;
                aval[2] =  1.0  * val0;
                asub[0] = sub_shift + (i+1) * s1d1CtrlP_num - 3;    
                asub[1] = sub_shift + (i+1) * s1d1CtrlP_num - 2;   
                asub[2] = sub_shift + (i+1) * s1d1CtrlP_num - 1;   
                // Next segment's first velocity control point
                aval[3] =  -1.0  * val1;
                aval[4] =   2.0  * val1;
                aval[5] =  -1.0  * val1;
                asub[3] = sub_shift + s1CtrlP_num + i * s1d1CtrlP_num;    
                asub[4] = sub_shift + s1CtrlP_num + i * s1d1CtrlP_num + 1;
                asub[5] = sub_shift + s1CtrlP_num + i * s1d1CtrlP_num + 2;

                r = MSK_putarow(task, row_idx, nzi, asub, aval);    
                row_idx ++;
            }

            sub_shift += s1CtrlP_num;
        }
    }


    /*******************************构造目标函数的对称矩阵Q(MtQM)*******************************/
    //ROS_WARN("[Bezier Trajectory] Start stacking the objective");
    
    int min_order_l = floor(minimize_order);
    int min_order_u = ceil (minimize_order);

    int NUMQNZ = 0;
    for(int i = 0; i < segment_num; i++)
    {
        int NUMQ_blk = (traj_order + 1);
        NUMQNZ += 3 * NUMQ_blk * (NUMQ_blk + 1) / 2; // 完整的Q矩阵中非零元素的个数. 例如7阶多项式时, 一段轨迹的一个轴Q矩阵元素个数为: 8*9/2=36
        // ROS_WARN("NUMQ_blk = %d", NUMQ_blk);
        // ROS_WARN("NUMQNZ = %d", NUMQNZ);
    }
    MSKint32t qsubi[NUMQNZ], qsubj[NUMQNZ];
    double qval[NUMQNZ];

    {    
        int sub_shift = 0;
        int idx = 0;
        for(int k = 0; k < segment_num; k ++)
        {
            double scale_k = corridor[k].t;
            for(int p = 0; p < 3; p ++ )
                for( int i = 0; i < s1d1CtrlP_num; i ++ )
                    for( int j = 0; j < s1d1CtrlP_num; j ++ )
                        if( i >= j )   // 只填充Q矩阵的下三角区域
                        {
                            // 将矩阵转换为3个数组, qsubi: 每个元素的行数, qsubj: 每个元素的列数, qval: 每个元素的值
                            qsubi[idx] = sub_shift + p * s1d1CtrlP_num + i;   
                            qsubj[idx] = sub_shift + p * s1d1CtrlP_num + j;  
                            // qval[idx]  = MQM(i, j) / (double)pow(scale_k, 3);
                            if(min_order_l == min_order_u)
                                qval[idx]  = MQM(i, j) / (double)pow(scale_k, 2 * min_order_u - 3);  // 1/s^(2*k-3)为比例系数, 可参见论文
                            else
                                qval[idx] = ( (minimize_order - min_order_l) / (double)pow(scale_k, 2 * min_order_u - 3)
                                            + (min_order_u - minimize_order) / (double)pow(scale_k, 2 * min_order_l - 3) ) * MQM(i, j);
                            idx ++ ;
                        }

            sub_shift += s1CtrlP_num;
        }
    }

    if ( r== MSK_RES_OK )
         r = MSK_putqobj(task,NUMQNZ,qsubi,qsubj,qval); 
    
    if ( r==MSK_RES_OK ) 
         r = MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE);


    ros::Time time_end1 = ros::Time::now();
    /*******************************使用Mosek求解QP问题*******************************/
    bool solve_ok = false;
    if (r == MSK_RES_OK)
    {
        //ROS_WARN("Prepare to solve the problem ");
        MSKrescodee trmcode;
        r = MSK_optimizetrm(task, &trmcode);
        MSK_solutionsummary(task, MSK_STREAM_LOG);

        if (r == MSK_RES_OK)
        {
            MSKsolstae solsta;
            MSK_getsolsta(task, MSK_SOL_ITR, &solsta);

            switch (solsta)
            {
            case MSK_SOL_STA_OPTIMAL:
            case MSK_SOL_STA_NEAR_OPTIMAL:
                // 获得最优值
                r = MSK_getxx(task,
                              MSK_SOL_ITR, // Request the interior solution.
                              x_var);

                // 获得最优时目标函数的值
                r = MSK_getprimalobj(
                    task,
                    MSK_SOL_ITR,
                    &primalobj);

                obj = primalobj;
                solve_ok = true;

                break;

            case MSK_SOL_STA_DUAL_INFEAS_CER:
            case MSK_SOL_STA_PRIM_INFEAS_CER:
            case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
            case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:
                printf("Primal or dual infeasibility certificate found.\n");
                break;

            case MSK_SOL_STA_UNKNOWN:
                printf("The status of the solution could not be determined.\n");
                //solve_ok = true; // debug
                break;
            default:
                printf("Other solution status.");
                break;
            }
        }
        else
        {
            printf("Error while optimizing.\n");
        }
    }

    if (r != MSK_RES_OK)
    {
        // In case of an error print error code and description.
        char symname[MSK_MAX_STR_LEN];
        char desc[MSK_MAX_STR_LEN];

        printf("An error occurred while optimizing.\n");
        MSK_getcodedesc(r,
                        symname,
                        desc);
        printf("Error %s - '%s'\n", symname, desc);
    }

    MSK_deletetask(&task);
    MSK_deleteenv(&env);

    ros::Time time_end2 = ros::Time::now();
    ROS_WARN("time consume in optimize is :");
    cout << time_end2 - time_end1 << endl;

    if (!solve_ok)
    {
        ROS_WARN("In solver, falied ");
        return -1;
    }

    /*******************************将求解结果(分段多项式系数)存入PolyCoeff中*******************************/
    VectorXd d_var(ctrlP_num);
    for (int i = 0; i < ctrlP_num; i++)
        d_var(i) = x_var[i];

    PolyCoeff = MatrixXd::Zero(segment_num, 3 * (traj_order + 1));

    int var_shift = 0;
    for (int i = 0; i < segment_num; i++)
    {
        for (int j = 0; j < s1CtrlP_num; j++)
            PolyCoeff(i, j) = d_var(j + var_shift);

        var_shift += s1CtrlP_num;
    }

    return 1;
}