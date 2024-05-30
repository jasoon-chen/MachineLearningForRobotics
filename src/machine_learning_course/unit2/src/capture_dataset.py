#! /usr/bin/env python

import rospy
from move_robot import MoveRobot
from nav_msgs.msg import Odometry
import os

class RobotGo():
    def __init__(self):

        #here we subscriber the data from "Odometry" which is going to call the callback function
        self.sub = rospy.Subscriber('/odom', Odometry, self.callback)
        #here we call the object from class MoveRobot (file move_robot.py)
        self.moverobot_object = MoveRobot()
        self.i = 0
        self.flag = False

    #call back function
    def callback(self, msg):
        # we capture the robot position x and y (xx,yy)
        xx = msg.pose.pose.position.x
        yy = msg.pose.pose.position.y
       # print the values o the terminal (shell) and save this data to the file
        if (self.flag == False):
            print xx, " :: ", yy
            file1.write("%f, %f\n" %((xx),(yy)))
        
        # we chose the linear and angular speed of the robot. Robot moves according to the set speeds
        linear_x = 0.25
        angular_z = 0.085
        self.moverobot_object.send_cmd(linear_x, angular_z)
        #we repeat the loop 300 times. We stop capturing data after the robot hits the wall
        self.i = self.i + 1
        if self.i > 300:
            self.flag = True
            linear_x = 0.0
            angular_z = 0.0
            self.moverobot_object.send_cmd(linear_x, angular_z)
            




if __name__ == '__main__':
    #we choose the place where to save the data
    try:
        os.mkdir('/home/user/catkin_ws/src/results')
    except:
        print "Folder already exist"
        pass
    file1 = open("/home/user/catkin_ws/src/results/test.csv", "w+")
    #defie the node and run the program/move the robot
    rospy.init_node('node')
    stopwall_object = RobotGo()
    rospy.spin()
