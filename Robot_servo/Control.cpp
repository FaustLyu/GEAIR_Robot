#include "Control.h"
#include "conf.h"

URControl::URControl(RTDEControlInterface *rtde_control, RTDEReceiveInterface *rtde_receive, Mat c2t_m, vector<vector<double>> path)
    : rtde_control_ptr(rtde_control), rtde_receive_ptr(rtde_receive), c2t_mat(c2t_m), position_path(path)
{
    init();
}
URControl::~URControl()
{
    stop_monitoring.store(true);
    if (monitor_thread.joinable())
    {
        monitor_thread.join();
    }
}
void URControl::init()
{
    current_take_photo_position = {0.1, -0.29, 0.530, 0.008, -2.099, 2.398};
    oneSideCameraFlower = {};
    stop_monitoring.store(false, std::memory_order_release);
    rot = true;
    flag.store(false, std::memory_order_release);
}
void URControl::run()
{
    CheckRobotMotionStateLoop();
    StartRobotService();
}
void URControl::StartRobotService()
{
    //while (true) {
    //   vector<double> pose =  rtde_receive_ptr->getActualTCPPose();
    //   vector<vector<double>> path = GetSmearPollenPath(pose);
    //   rtde_control_ptr->moveL(path, true);
    //   // 伺服抹粉
    //   Sleep(60000);
    //}
    int index_g = 0;
    while (true)
    {

        bool flag_success = sendSocketMessage("get_flag");
        if (flag_success)
        {
            flag.load(std::memory_order_acquire);
            if (flag)
            {
                std::vector<std::vector<double>> left_tcp_pos;
                left_tcp_pos.assign(position_path.begin(), position_path.begin() + LEFT_SIZE - 1);
                TakeOneSidePhotos(left_tcp_pos);
                std::cout << "准备左侧开始插值" << std::endl;
                // 插值，让机械臂能顺利到达左边的授粉点
                do
                {
                    InsertMoveJ(rtde_control_ptr,3,-1.3);
                } while (false);
                do
                {
                    InsertMoveJ(rtde_control_ptr,1,0.5);
                } while (false);
                // 移动到左边授粉点
                MoveToTarget(position_path[LEFT_SIZE - 1]);
                pollination_start_positon = position_path[LEFT_SIZE - 1];
                // 从授粉点开始授粉
                StartPollinate_new(oneSideCameraFlower);
                ClearCameraPositions();
                // 机械臂打下去，准备转过去
                do
                {
                    InsertMoveJ(rtde_control_ptr,1,-0.5);
                } while (false);

                //-------------
                


                //------------

                // 转过右边
                do
                {
                    InsertMoveJ(rtde_control_ptr,0,ROT_ANGLE);
                } while (false);

                do
                {
                    InsertMoveJ(rtde_control_ptr, 1, -0.8);
                } while (false);
                std::vector<std::vector<double>> right_tcp_pos;
                right_tcp_pos.assign(position_path.begin() + LEFT_SIZE, position_path.end() - 1);
                TakeOneSidePhotos(right_tcp_pos);
                // 插值，让机械臂顺利到达右边授粉开始点，未完。。。旋转的角度要实验后再定
                do
                {
                    InsertMoveJ(rtde_control_ptr,3,-1.3);
                } while (false);
                do
                {
                    InsertMoveJ(rtde_control_ptr, 2, 0.5);
                } while (false);
                do
                {
                    InsertMoveJ(rtde_control_ptr,1,0.6);
                } while (false);
                // 移动到右边授粉点
                MoveToTarget(position_path[position_path.size() - 1]);
                pollination_start_positon = position_path[position_path.size() - 1];
                // 从授粉点开始授粉
                StartPollinate_new(oneSideCameraFlower);
                ClearCameraPositions();
                do
                {
                    InsertMoveJ(rtde_control_ptr,1,-0.5);
                } while (false);
                // 机械臂转回左边
                do
                {
                    InsertMoveJ(rtde_control_ptr,0,-ROT_ANGLE);
                } while (false);
                // 插值让机械臂能够顺利到达左边第一授粉点
                do
                {
                    InsertMoveJ(rtde_control_ptr,1, -0.5);
                } while (false);
                bool update_flag_success = sendSocketMessage("update_flag");
                if (update_flag_success)
                    cout << "发送更改小车状态命令成功" << endl;
                flag.store(false, std::memory_order_release);
            }
            else
            {
                cout << "no do anything" << endl;
            }
        }
        std::this_thread::sleep_for(3000ms);
    }
}
void URControl::TakeOneSidePhotos(std::vector<std::vector<double>> tcp_positions)
{
    for (std::vector<double> p : tcp_positions)
    {
        current_take_photo_position = p;
        MoveToTarget(current_take_photo_position);
        CallForPhoto();
        while (Wait_For_Photo.load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(200ms);
        }
    }
}
void URControl::CallForPhoto()
{
    Wait_For_Photo.store(true, std::memory_order_release);
    PrepareTakePhoto();
    bool success = sendSocketMessage("take_photo");
    if (!success)
    {
        Wait_For_Photo.store(false, std::memory_order_release);
    }
}
void URControl::MoveToTarget(vector<double> target)
{
    SetTargetPosition(target);
    RobotMoving.store(true, std::memory_order_release);
    rtde_control_ptr->moveL(target, MOVESPEED, ACCELERATION);
    while (RobotMoving.load(std::memory_order_acquire))
    {
        std::cout << "机械臂正在运动..." << std::endl;
        std::this_thread::sleep_for(100ms);
    }
    std::this_thread::sleep_for(500ms);
    std::cout << "机械臂运动完成" << std::endl;
}
void URControl::SetTargetPosition(const std::vector<double> &new_position)
{
    std::lock_guard<std::mutex> lock(position_mutex);
    target_postion = new_position;
}
vector<double> URControl::GetTargetPosition()
{
    std::lock_guard<std::mutex> lock(position_mutex);
    return target_postion; // 返回的是一个拷贝
}
void URControl::CheckRobotMotionStateLoop()
{
    // 启动异步线程
    monitor_thread = std::thread([this]()
                                 {
        while (!stop_monitoring.load(std::memory_order_acquire)) {
            if (RobotMoving.load(std::memory_order_acquire)) {
                vector<double> postion = rtde_receive_ptr->getActualTCPPose();
                double position_error = 0.0;
                vector<double> target_position = GetTargetPosition();
                for (size_t i = 0; i < 3; ++i) {
                    position_error += std::pow(target_position[i] - postion[i], 2);
                }
                position_error = std::sqrt(position_error);
                if (position_error < ROBOTMOVEERROR) {
                    RobotMoving.store(false, std::memory_order_release);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待100毫秒
        } });
}
void URControl::Pollinate(RTDEControlInterface *rtde_control, vector<double> tcp, vector<double> camera_positions)
{
    vector<double> vWorldP = Camera2Tcp(camera_positions, c2t_mat);
    std::cout << "vWorldP vWorldP vWorldP" << std::endl;
    PrintVct(vWorldP);
    std::cout << "tcp tcp tcp" << std::endl;
    PrintVct(tcp);
    vector<double> base = Tcp2Base(tcp, vWorldP);
    std::cout << "base base base" << std::endl;
    PrintVct(base);
    std::vector<double> startPoint(current_take_photo_position);
    startPoint[0] = base[0];
    startPoint[1] = base[1];
    startPoint[2] = base[2];

    if (sqrt(startPoint[0] * startPoint[0] + startPoint[1] * startPoint[1] + startPoint[2] * startPoint[2]) <= 0.8)
    {
        if (startPoint[0] >= -0.5 && startPoint[0] <= 0.5 &&
            startPoint[1] >= -0.6 && startPoint[1] <= 0.6 &&
            startPoint[2] > 0 && startPoint[2] <= 0.7)
        {
            MoveToTarget(startPoint);
            vector<vector<double>> path = GetSmearPollenPath(startPoint);
            rtde_control->moveL(path, true);
            // 伺服抹粉
            Sleep(12000);
        }
        else
        {
            cout << "局部超出安全距离！" << endl;
        }
    }
    else
    {
        cout << "整体超出安全距离！" << endl;
    }
}
void URControl::Pollinate_new(RTDEControlInterface *rtde_control, vector<double> pollinationStart, vector<double> base)
{
    std::vector<double> startPoint(base);

    if (startPoint[0] >= -0.25 && startPoint[0] <= 0.25 &&
        abs(startPoint[1]) >= 0.19 && abs(startPoint[1]) <= 0.60 &&
        startPoint[2] > 0 && startPoint[2] <= 0.90)
    {

        std::cout << "开始到达target：" << std::endl;
        PrintVct(startPoint);
        MoveToTarget(startPoint);
        vector<vector<double>> path = GetSmearPollenPath(startPoint);
        if (IS_HAIR == 1)
        {
            sendSocketMessage("open_switch");
        }
        rtde_control->moveL(path, true);
        // 伺服抹粉
        Sleep(25000);

        if (IS_HAIR == 1) {
            sendSocketMessage("close_switch");
        }

    }
    
    if (sqrt(startPoint[0] * startPoint[0] + startPoint[1] * startPoint[1] + startPoint[2] * startPoint[2]) <= 0.85)
    {
        if (startPoint[0] >= -0.3 && startPoint[0] <= 0.3 &&
            startPoint[1] >= -0.7 && startPoint[1] <= 0.7 &&
            startPoint[2] > 0 && startPoint[2] <= 0.85)
        {
            std::vector<double> middlePoint(pollinationStart);
            middlePoint[1] += startPoint[1] > 0 ? 0.15 : -0.15;
            MoveToTarget(middlePoint);
            std::cout << "开始到达target：" << std::endl;
            PrintVct(startPoint);
            MoveToTarget(startPoint);
            vector<vector<double>> path = GetSmearPollenPath(startPoint);
            rtde_control->moveL(path, true);
            Sleep(12000);
        }
        else
        {
            cout << "局部超出安全距离！" << endl;
        }
    }
    else
    {
        cout << "整体超出安全距离！" << endl;
    }
}
void URControl::StartPollinate(vector<vector<double>> camera_positions)
{
    for (int i = 0; i < camera_positions.size(); i++)
    {
        vector<double> camera_position = camera_positions[i];
        Pollinate(rtde_control_ptr, current_take_photo_position, camera_position);
        MoveToTarget(current_take_photo_position);
    }
    Wait_For_Photo.store(false, std::memory_order_release);
}
void URControl::setCallback(std::function<bool(const char *)> callback)
{
    sendSocketMessage = callback;
}
void URControl::PrepareTakePhoto()
{
    vector<double> postion = rtde_receive_ptr->getActualTCPPose();
    current_take_photo_position = postion;
}
void URControl::receiveMessage(Json::Value data)
{
    bool is_number = false;
    // 创建 Json::Reader 对象
    Json::CharReaderBuilder readerBuilder;
    Json::CharReader *reader = readerBuilder.newCharReader();
    Json::Value jsonObject;
    std::string errs;

    std::string dataString = data.asString();
    // 解析 "data" 字段中的嵌套 JSON 数组
    Json::Value jsonArray;
    bool parsingSuccessful = reader->parse(dataString.c_str(), dataString.c_str() + dataString.size(), &jsonArray, &errs);

    if (!parsingSuccessful)
    {
        std::cout << "Failed to parse the nested JSON string: " << errs << std::endl;
        return;
    }

    // 检查是否为数组
    if (!jsonArray.isArray())
    {
        std::cout << "The nested JSON string is not an array." << std::endl;
        return;
    }
    std::vector<std::vector<double>> arr;
    // 遍历 JSON 数组
    for (const auto &item : jsonArray)
    {
        if (item.isObject())
        {
            double x = item["x"].asDouble();
            double y = item["y"].asDouble();
            double z = item["z"].asDouble();
            if(IS_HAIR == 1) z -= 0.05;

            if (x != 0 && y != 0 && z > 0)
            {
                vector<double> camera_positions = {x, y, z};
                arr.push_back(camera_positions);
            }
        }
        else
        {
            cout << item << endl;
            is_number = true;
            item == 0 ? flag.store(false, std::memory_order_release) : flag.store(true, std::memory_order_release);
        }
    }
    if (!is_number)
        AddCameraPositions(arr);
    Wait_For_Photo.store(false, std::memory_order_release);
}
void URControl::AddCameraPositions(vector<vector<double>> positions)
{

    for (int i = 0; i < positions.size(); i++)
    {
        vector<double> camera_position = positions[i];
        vector<double> vWorldP = Camera2Tcp(camera_position, c2t_mat);
        vector<double> base = Tcp2Base(current_take_photo_position, vWorldP);
        std::vector<double> startPoint(current_take_photo_position);
        startPoint[0] = base[0];
        startPoint[1] = base[1];
        startPoint[2] = base[2];
        oneSideCameraFlower.push_back(startPoint);
    }
}
void URControl::ClearCameraPositions()
{
    oneSideCameraFlower.clear();
}
void URControl::StartPollinate_new(vector<vector<double>> tcp_positons)
{
    std::cout << "开始授粉： " << tcp_positons.size() << std::endl;
    for (int i = 0; i < tcp_positons.size(); i++)
    {
        vector<double> pos = tcp_positons[i];
        Pollinate_new(rtde_control_ptr, pollination_start_positon, pos);
        MoveToTarget(pollination_start_positon);
    }
}
void URControl::InsertMoveJ(RTDEControlInterface *rtde_control, int joint, double value)
{
    std::vector<double> joint_angles = rtde_receive_ptr->getActualQ();
    joint_angles[joint] += value;
    rtde_control->moveJ(joint_angles, JOINT_ANGLE_MOVESPEED, ACCELERATION);
    std::this_thread::sleep_for(1000ms);
}
