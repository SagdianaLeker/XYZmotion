import matplotlib.pyplot as plt
import numpy as np
import time

class Robot:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])  # initial position (x, y, z)
        self.positions = [self.position.copy()]
        self.fig, self.ax = self._init_plot()

    def _init_plot(self):
        plt.ion()  # turn on interactive mode
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def move(self, axis, distance): # move by specified axis
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis in axis_map:
            axis_index = axis_map[axis]
            current_position = self.position[axis_index]
            target_position = current_position + distance
            steps = 5
            for step in np.linspace(current_position, target_position, steps):
                self.position[axis_index] = step
                self.positions.append(self.position.copy())
                self.real_time_visualization()
                time.sleep(0.01)
            
        else:
            raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    def set_position(self, x, y, z): # set target postion 
        target_position = np.array([x, y, z])
        current_position = self.position.copy()

        steps = 5
        for axis in range(3):
            for step in np.linspace(current_position[axis], target_position[axis], steps):
                self.position[axis] = step
                self.positions.append(self.position.copy())
                self.real_time_visualization()
                time.sleep(0.01)  

    def get_positions(self):
        return np.array(self.positions)

    def real_time_visualization(self):
        positions = self.get_positions()

        self.ax.clear()
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Path', linestyle='--')
        self.ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', label='Start')
        self.ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', label='Current Position')

        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.set_title('Real-Time XYZ Movement Path')
        self.ax.legend()

        plt.draw()
        plt.pause(0.02)

def main():
    turtle = Robot()

    print("Robot initialized. Enter commands to control the system.")
    print("Commands:")
    print("  move <axis> <distance>   - Move along axis (x, y, z) by a distance")
    print("  set <x> <y> <z>          - Set position to specified coordinates with smooth motion")
    print("  exit                     - Exit the program")

    while True:
        user_input = input("Enter command: ").strip().split()
        if not user_input:
            continue

        command = user_input[0].lower()

        if command == 'move':
            if len(user_input) != 3:
                print("Invalid move command. Usage: move <axis> <distance>")
                continue
            axis, distance = user_input[1], float(user_input[2])
            try:
                turtle.move(axis, distance)
                print(f"Moved along {axis} by {distance}. Current position: {turtle.position}")
            except ValueError as e:
                print(e)

        elif command == 'set':
            if len(user_input) != 4:
                print("Invalid set command. Usage: set <x> <y> <z>")
                continue
            x, y, z = map(float, user_input[1:4])
            turtle.set_position(x, y, z)
            print(f"Position set to: {turtle.position}")

        elif command == 'exit':
            print("Exiting program.")
            break

        else:
            print("Unknown command. Please try again.")

    plt.ioff()  # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()

