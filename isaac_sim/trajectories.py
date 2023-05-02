#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import math

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

class Trajectory:

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
        	desired duration of the trajectory in seconds 
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        
        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0], 
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0], 
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0], 
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0], 
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0], 
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0], 
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):

    def __init__(self, start, goal, time):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, time)

        self.start = start
        self.goal = goal

        self.dist = np.linalg.norm(self.goal - self.start)
        self.dir = (self.goal - self.start) / self.dist

        self.max_vel = self.dist / (self.total_time * .8)
        self.m = -self.max_vel / (.2 * self.total_time)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pos_goal = 0
        tfrac = time/self.total_time

        if tfrac <= .2:
            pos_goal = self.max_vel / (.4 * self.total_time) * time ** 2
        elif tfrac <= .8:
            pos_goal = self.max_vel * (time - .2 * self.total_time) + .125 * self.dist
        elif tfrac <= 1:
            pos_goal = self.m * (time ** 2 / 2 - time * self.total_time - .32 * self.total_time ** 2 + .8 * self.total_time ** 2) + .875 * self.dist
        else:
            pos_goal = self.dist

        return np.concatenate((self.start + pos_goal * self.dir, [0, 1, 0, 0]), axis=None)

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        vel_goal = 0
        tfrac = time/self.total_time

        if time > 0:
            if tfrac <= .2:
                vel_goal = self.max_vel/(.2 * self.total_time) * time
            elif tfrac <= .8:
                vel_goal = self.max_vel
            elif tfrac <= 1:
                vel_goal = self.m * (time - self.total_time)

        return np.concatenate((vel_goal * self.dir, [0, 0, 0]), axis=None)



class CircularTrajectory(Trajectory):

    def __init__(self, center_position, radius, total_time):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, total_time)
        self.center_position = center_position
        self.radius = radius
        self.dist = 2 * np.pi

        self.max_vel = self.dist / (self.total_time * .8)
        self.m = -self.max_vel / (.2 * self.total_time)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """

        angle = self.get_theta(time)
        pos_x = self.radius * np.cos(angle) + self.center_position[0]
        pos_y = self.radius * np.sin(angle) + self.center_position[1]
        return np.array([pos_x,pos_y,0,0,1,0,0])

    def get_theta(self, time):
        angle = 0
        tfrac = time/self.total_time

        if tfrac <= .2:
            angle = self.max_vel / (.4 * self.total_time) * time ** 2
        elif tfrac <= .8:
            angle = self.max_vel * (time - .2 * self.total_time) + .125 * self.dist
        elif tfrac <= 1:
            angle = self.m * (time ** 2 / 2 - time * self.total_time - .32 * self.total_time ** 2 + .8 * self.total_time ** 2) + .875 * self.dist
        else:
            angle = self.dist

        return angle


    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        vel_goal = 0
        tfrac = time/self.total_time

        if time > 0:
            if tfrac <= .2:
                vel_goal = self.max_vel/(.2 * self.total_time) * time
            elif tfrac <= .8:
                vel_goal = self.max_vel
            elif tfrac <= 1:
                vel_goal = self.m * (time - self.total_time)
        theta = self.get_theta(time)
        dir = np.array([-np.sin(theta), np.cos(theta), 0])
        return np.concatenate((self.radius * vel_goal * dir, [0, 0, 0]), axis=None)
        pass

class PolygonalTrajectory(Trajectory):
    def __init__(self, points, total_time):
        """
        Remember to call the constructor of Trajectory.
        You may wish to reuse other trajectories previously defined in this file.

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit

        """
        self.sub_time = total_time/len(points)
        Trajectory.__init__(self, total_time)
        self.points = np.concatenate((points, [points[0]]), axis=0)

    def update_val(self, time):
        index = math.floor(time/self.sub_time)
        if index > len(self.points) - 2:
            index = len(self.points) - 2

        self.start = self.points[index]
        self.goal = self.points[index + 1]

        self.dist = np.linalg.norm(self.goal - self.start)
        self.dir = (self.goal - self.start) / self.dist

        self.max_vel = self.dist / (self.sub_time * .8)
        self.m = -self.max_vel / (.2 * self.sub_time)

        return time - index * self.sub_time


    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        time = self.update_val(time)
        pos_goal = 0
        tfrac = time/self.sub_time

        if tfrac <= .2:
            pos_goal = self.max_vel / (.4 * self.sub_time) * time ** 2
        elif tfrac <= .8:
            pos_goal = self.max_vel * (time - .2 * self.sub_time) + .125 * self.dist
        elif tfrac <= 1:
            pos_goal = self.m * (time ** 2 / 2 - time * self.sub_time - .32 * self.sub_time ** 2 + .8 * self.sub_time ** 2) + .875 * self.dist
        else:
            pos_goal = self.dist

        return np.concatenate((self.start + pos_goal * self.dir, [0, 1, 0, 0]), axis=None)
        
    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        time = self.update_val(time)
        vel_goal = 0
        tfrac = time/self.sub_time

        if time > 0:
            if tfrac <= .2:
                vel_goal = self.max_vel/(.2 * self.sub_time) * time
            elif tfrac <= .8:
                vel_goal = self.max_vel
            elif tfrac <= 1:
                vel_goal = self.m * (time - self.sub_time)

        return np.concatenate((vel_goal * self.dir, [0, 0, 0]), axis=None)


def define_trajectories(args):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory(np.array([0,10,0]), np.array([2,10,0]), 5)
    elif args.task == 'circle':
        trajectory = CircularTrajectory(np.array([0,10,0]),5,5)
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory(np.array([[0, 10, 0],[10,0,0],[0,0,10],[10,10,10]]), 15)
    return trajectory

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)
    
    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)
