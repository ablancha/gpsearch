import numpy as np
import dubins


class PathPlanner(object):
    """A class for path parametrization using Dubins curves.

    Parameters
    ---------
    domain : list
        Domain definition.  Must be of the form 
            [ [xmin, xmax], [ymin, ymax] ]
    turning_radius : float, optional
        Turning radius for Dubins curve.
    look_ahead : float, optional
        Robot's horizon. Candidate destinations lie on a circle
        of radius `look_ahead` centered at the current pose.
    n_frontier : integrer, optional
        Number of candidate destinations considered by the robot.
    fov : integer, optional
        Robot's field of vision. Excludes candidate destinations that
        lie outside the interval [-fov, fov] (behind the robot).
    padding : float, optional
        Minimum allowable distance to the domain boundaries. Candidate
        destinations that are less than `padding` away are discarded.
        Default is `2*turning_radius`.

    Attributes
    ----------
    domain, turning_radius, look_ahead, n_frontier, fov, 
    padding : see Parameters

    """

    def __init__(self, domain, turning_radius=0.01, look_ahead=0.1, 
                 n_frontier=100, fov=0.75*np.pi, padding=None):
        self.domain = domain
        self.turning_radius = turning_radius
        self.look_ahead = look_ahead
        self.n_frontier = n_frontier
        self.fov = fov
        if padding is None:
            padding = 2*turning_radius
        self.padding = padding

    def make_frontier(self, q_pose):
        """Construct list of candidate destinations for the current pose.

        Parameters
        ----------
        q_pose : array_type
            Current pose in `(x,y,angle)` format.
      
        Returns
        -------
        targets : list
            A list of candidate poses for the robot to pick from.

        """
        x_pose, y_pose, a_pose = q_pose
        a_target = a_pose + np.linspace(-self.fov, self.fov, 
                                        self.n_frontier) 
        x_target = x_pose + self.look_ahead * np.cos(a_target)
        y_target = y_pose + self.look_ahead * np.sin(a_target)
        frontier = np.vstack((x_target,y_target,a_target)).T
        [x_min, x_max], [y_min, y_max] = self.domain
        retain = [ ( ff[0] >= x_min + self.padding and 
                     ff[0] <= x_max - self.padding and
                     ff[1] >= y_min + self.padding and 
                     ff[1] <= y_max - self.padding 
                   ) for ff in frontier ]
        targets = frontier[retain].tolist()
        return targets

    def make_paths(self, q_pose):
        """Construct a list of Dubins curve between current pose and 
        candidate destinations.

        Parameters
        ----------
        q_pose : array_type
            Current pose in `(x,y,angle)` format.
      
        Returns
        -------
        paths : list of Dubins `path` instances
            A list of candidate paths between current pose and 
            candidate destinations.

        """
        frontier = self.make_frontier(q_pose)
        paths = [ dubins.shortest_path(q_pose, q_target, 
                                       self.turning_radius)
                  for q_target in frontier ]
        return paths

    def make_itinerary(self, path, sample_size):
        """Generate a list of poses sampled equidistantly along
        a Dubins path.  
        
        Parameters
        ----------                     
        path : instance of Dubins `path`
            A Dubins path.
        sample_size : integer, optional
            How many (equidistant) poses are samples along the path. 
      
        Returns
        -------
        samples : list of tuples
            A list of `sample_size` poses from the Dubins path.

        """
        step_size = path.path_length()/sample_size - 1e-10
        samples = path.sample_many(step_size)
        return samples
    
