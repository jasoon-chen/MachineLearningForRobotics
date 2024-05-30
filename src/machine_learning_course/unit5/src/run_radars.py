#! /usr/bin/env python

import rospy
from move_robot_for_laser_scan import MoveRobot
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import os
import numpy as np
import time



def pol2cart(rho, phi):
    """
    Method to transform from polar to cartesion system
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


class RobotGo():


    def __init__(self):
        """
        Class constructor
        """
        self.sub = rospy.Subscriber('/odom', Odometry, self.callback)
        self.sub2 = rospy.Subscriber('/kobuki/laser/scan', LaserScan, self.call_laser)
        self.moverobot_object = MoveRobot()
        self.robot_theta = 0
        self.radar1 = 0
        self.radar2 = 0
        self.x1 = 0
        self.y1 = 0 
        self.x2 = 0
        self.y2 = 0



    def call_laser(self, msg):
        """
        Definition of callback method to return information about laser scan - position of detected objects
        """

        self.radar1 = msg.ranges[360]
        self.radar2 = msg.ranges[400]
       

    def callback(self, msg):
        """
        Definition of callback method to move the robot and save robot position and laser values to file
        """

        ori_z = msg.pose.pose.orientation.z
        rot_q = msg.pose.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        #speed of the robot movement
        linear_x = 0.0
        angular_z = 0.4 
        self.moverobot_object.send_cmd(linear_x, angular_z)

        radar1 = self.radar1
        radar2 = self.radar2 
        #x1, y1, x2, y2 = 0

        # we save only values (from laser) between 0 and 10 meters 
        if radar1 < float(10) and radar1 > float (0.0):
            self.x1 , self.y1 = pol2cart(radar1, theta)
            #self.x1 = xx1
            #self.y1 = yy1 
        if radar2 < float(10) and radar2 > float (0.0):
            self.x2 , self.y2 = pol2cart(radar2, theta)
            #self.x2 = xx2
            #self.y2 = yy2 

        if self.x1 != 0 and self.y1 != 0 and self.x2 != 0 and self.y2 != 0:
            file1.write("%f,%f,%f, %f\n" %((self.x1), (self.y1),  (self.x2), (self.y2)))
            print "radar1 :: ", self.x1 , " :: ", self.y1, " radar2 :: ", self.x2, " :: ", self.y2


    
if __name__ == '__main__':
    """
    The program starts here
    """
    try:
        os.mkdir('/home/user/catkin_ws/src/results')
    except:
        print "Folder already exist"
        pass
    file1 = open("/home/user/catkin_ws/src/results/radar_1_2.csv", "w+")
    rospy.init_node('node')
    stopwall_object = RobotGo()
    #we keep alive callback functions
    rospy.spin()
