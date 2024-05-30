#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf import transformations
from tf.transformations import euler_from_quaternion
import numpy as np
import math

pub_ = None
radar_front_ =0
state_ = 0
state_dict_ = {
    0: 'detect the wall',
    1: 'turn left',
    2: 'turn right',
}



def load_weights():
    
    W2 = np.load('/home/user/catkin_ws/src/machine_learning_course/weights_demo/w2.npy')
    b2 = np.load('/home/user/catkin_ws/src/machine_learning_course/weights_demo/b2.npy')
    W = np.load('/home/user/catkin_ws/src/machine_learning_course/weights_demo/w.npy')
    b = np.load('/home/user/catkin_ws/src/machine_learning_course/weights_demo/b.npy')
    
    return W2,b2,W,b


W2,b2,W,b = load_weights()



def test_prediction(XX):
    
    W2,b2,W,b = load_weights()
    hidden_layer = np.maximum(0, np.dot(XX, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    prediction = np.argmax(scores, axis=1)
    return int(prediction)

 
def clbk_laser(msg):

    global radar_front_
    radar_front_ =  min(min(msg.ranges[320:400]), 10) 
    take_action()
   

def change_state(state):

    global state_, state_dict_
    if state is not state_:
        print 'Wall follower - [%s] - %s' % (state, state_dict_[state])
        state_ = state


def take_action():
    
    global radar_front_
    radar_front = radar_front_
    msg = Twist()
    linear_x = 0
    angular_z = 0
    state_description = ''
   
    XX = np.asarray([0,radar_front])
    pred = test_prediction(XX)

    """
    The state machine of our robot
    """
    if pred == 0:
        change_state(0)
    if pred == 1:
        change_state(2)
    
"""
Motion functions
"""
def find_wall():
    msg = Twist()
    msg.linear.x = 0.15 
    msg.angular.z = 0.0 
    return msg


def turn_right():
    msg = Twist()
    msg.linear.x = 0.0
    msg.angular.z = -1 
    return msg

def stop():
    msg = Twist()
    msg.linear.x = 0.0
    msg.angular.z = 0.0 
    for x in range(2):
        pub_.publish(msg)
        rospy.loginfo('Stopping robot')


def main():
    global pub_

    rospy.init_node('reading_laser')
    rospy.on_shutdown(stop)
    pub_ = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    sub = rospy.Subscriber('/kobuki/laser/scan', LaserScan, clbk_laser)

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        msg = Twist()
        if state_ == 0:
            msg = find_wall()

        elif state_ == 2:
            msg = turn_right()
            pass
        else:
            rospy.logerr('Unknown state!')

        pub_.publish(msg)

        rate.sleep()
    
    


if __name__ == '__main__':
    main()
