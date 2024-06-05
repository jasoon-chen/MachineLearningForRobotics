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


roll_ = 0
pitch_ = 0
theta_ = 0

radar_front_ = 0
radar_left_ = 0
radar_right_ = 0

state_ = 0

state_dict_ = {
    0: 'detect the wall',
    1: 'turn left',
    2: 'turn right',
}



"""
## YOUR TASK NR: 1
Define the function, which loads the weights and biases
HINT: Reuse the same code for neural network you use in Unit 3 of this course
HINT: check the place where are your training weights (Jupyter Notebook)
"""

def load_weights():
    W2 = np.load('/home/user/catkin_ws/src/weights_proj/w2.npy')
    b2 = np.load('/home/user/catkin_ws/src/weights_proj/b2.npy')
    W = np.load('/home/user/catkin_ws/src/weights_proj/w.npy')
    b = np.load('/home/user/catkin_ws/src/weights_proj/b.npy')

    return W2,b2,W,b

W2,b2,W,b = load_weights()


def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]



"""
## YOUR TASK NR: 2
Implement prediction function
HINT: Reuse the same code for neural network you use in Unit 3 of this course
"""

def test_prediction(XX):
    W2,b2,W,b = load_weights()
    hidden_layer = np.maximum(0, np.dot(XX, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    prediction = np.argmax(scores, axis=1)
    return int(prediction)

"""
## YOUR TASK NR: 3
Implement Method to transform from polar to cartesion system
HINT: Reuse the same code presented in Unit 4 of this course
"""

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)



def clbk_laser(msg):
   

    global radar_front_
    global radar_left_
    global radar_right_

    radar_front_ = min(min(msg.ranges[320:400]), 2)
    radar_left_ = min(msg.ranges[719], 10)
    radar_right_ = min(msg.ranges[1], 10)

    take_action()


def callback(msg):
    
    global roll_
    global pitch_
    global theta_

    
    ori_z = msg.pose.pose.orientation.z
    rot_q = msg.pose.pose.orientation
    (roll_, pitch_, theta_) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    

def change_state(state):
    global state_, state_dict_
    if state is not state_:
        print('Wall follower - [%s] - %s' % (state, state_dict_[state]))
        state_ = state


def take_action():
    global radar_front_
    global radar_left_
    global radar_right_
    
    global roll_
    global pitch_
    global theta_


    
    radar_front = radar_front_
    radar_left = radar_left_
    radar_right = radar_right_

    theta = theta_

    msg = Twist()
    linear_x = 0
    angular_z = 0

    state_description = ''

    d = 1.0 

    """
    Here we normalize the data form the radar 
    """
    theta_norm = normalize([theta],{'actual':{'lower':-3.1415,'upper':3.1415},'desired':{'lower':-1.3962,'upper':1.3962}})
    theta_norm = theta_norm[0]
    y1 , x1 = pol2cart(radar_front, theta_norm)


    x1_norm = normalize([x1],{'actual':{'lower':-2,'upper':2},'desired':{'lower':-1,'upper':1}})
    x1_norm = x1_norm[0]


    XX = np.asarray([x1_norm,abs(y1)])
    XX = np.asarray([x1_norm,radar_front])
    pred = test_prediction(XX)




    """
    The state machine of our robot
    """
    if pred == 0:
        change_state(0)

    if pred == 1 and (radar_right < radar_left): 
        change_state(1)
    
    if pred == 1 and (radar_right > radar_left): 
        change_state(2)   


"""
Motion functions
"""

def find_wall():
    msg = Twist()
    msg.linear.x = 0.15 
    msg.angular.z = 0.0 
    return msg


def turn_left():
    msg = Twist()

    msg.linear.x = 0.0
    msg.angular.z =5.0 
    return msg

def turn_right():
    msg = Twist()

    msg.linear.x = 0.0
    msg.angular.z = -5.0 
    return msg

"""
## YOUR TASK NR: 5
In while loop replace None for the correct number of state the robot performs the task (the states changes after
the laser detects object or not).
HINT: See the definition of state_dict in this script
""" 


def main():
    global pub_

    rospy.init_node('reading_laser')

    pub_ = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    
    sub2 = rospy.Subscriber('/odom', Odometry, callback)
    sub = rospy.Subscriber('/kobuki/laser/scan', LaserScan, clbk_laser)

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        msg = Twist()
        if state_ == 0:
            msg = find_wall()
        elif state_ == 1:
            msg = turn_left()
        elif state_ == 2:
            msg = turn_right()
            pass
        else:
            rospy.logerr('Unknown state!')

        pub_.publish(msg)

        rate.sleep()


if __name__ == '__main__':
    main()
