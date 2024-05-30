#! /usr/bin/env python

import rospy
from move_robot_NN import MoveRobot
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import os
import numpy as np
import time

def load_weights():
    
    W2 = np.load('/home/user/catkin_ws/src/weights/w2.npy')
    b2 = np.load('/home/user/catkin_ws/src/weights/b2.npy')
    W = np.load('/home/user/catkin_ws/src/weights/w.npy')
    b = np.load('/home/user/catkin_ws/src/weights/b.npy')
    
    return W2,b2,W,b


W2,b2,W,b = load_weights()


class RobotGo():
    def __init__(self):

        self.sub = rospy.Subscriber('/odom', Odometry, self.callback)
        self.sub2 = rospy.Subscriber('/kobuki/laser/scan', LaserScan, self.callback2)
        self.moverobot_object = MoveRobot()
        self.i = 0
        self.flag = False
        self.direction = 1
        self.prediction = 0
        self.radar = 0

    def pred (self, xx, yy):

        yy_n = (yy+4)/8
        xx_n = (xx+1)/7
        XX = np.asarray([xx_n,yy_n])
        hidden_layer = np.maximum(0, np.dot(XX, W) + b)
        scores = np.dot(hidden_layer, W2) + b2
        self.prediction = int(np.argmax(scores, axis=1))

        return self.prediction

    def callback2(self, msg):
        #print " "
        #print msg.ranges[360]
        self.radar = msg.ranges[360]


    def callback(self, msg):
        temp_pred = 0

        xx = msg.pose.pose.position.x
        yy = msg.pose.pose.position.y
        ori_z = msg.pose.pose.orientation.z
        yy_n = (yy+4)/8
        xx_n = (xx+1)/7
        print "xx :: ", xx , "yy :: ", yy



        XX = np.asarray([xx_n,yy_n])


        hidden_layer = np.maximum(0, np.dot(XX, W) + b)
        scores = np.dot(hidden_layer, W2) + b2
        pred = int(np.argmax(scores, axis=1))



        print"prediction", pred
        #print "radar :::", self.radar

        if (pred == 0):

            linear_x = 0.45
            angular_z = 0.0 #0.0 ok
            self.moverobot_object.send_cmd(linear_x, angular_z)



        if (pred == 1) :

            linear_x = -0.25
            angular_z = 0.0
            while (self.radar<2.5):
                print "distance from wall (waiting for >2.5) ::: ", self.radar
                self.moverobot_object.send_cmd(linear_x, angular_z)
            


if __name__ == '__main__':
   

    rospy.init_node('node')
    stopwall_object = RobotGo()
    rospy.spin()
