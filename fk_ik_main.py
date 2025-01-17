import numpy as np
import plotly.graph_objects as go
import p_fkdh as fk

# persistent state to save the last position and joint angles
current_joint_angles = [90, 60, -90]  # initial joint angles: J1, J2, J3
current_position = [0, 0, 0]  # initial end-effector position (X, Y, Z)

class PlotlyKinematicsVisualization:
    def __init__(self):
        self.fig = go.Figure()

    def update_plot(self, positions):
        """Update the 3D plot with the current positions."""
        X, Y, Z = positions
        self.fig.data = []  # clear previous data
        self.fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='lines+markers',
            marker=dict(size=5, color='blue'),
            line=dict(width=3, color='red')
        ))
        self.fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'  # ensures equal scaling
            ),
            title="Robotic Arm Visualization"
        )
        self.fig.show()

def run_fk(viz):
    """Perform forward kinematics and update the visualization."""
    global current_joint_angles, current_position

    print("\n--- Forward Kinematics ---")
    j1 = float(input(f"Enter Joint 1 angle (degrees) [current: {current_joint_angles[0]}]: ") or current_joint_angles[0])
    j2 = float(input(f"Enter Joint 2 angle (degrees) [current: {current_joint_angles[1]}]: ") or current_joint_angles[1])
    j3 = float(input(f"Enter Joint 3 angle (degrees) [current: {current_joint_angles[2]}]: ") or current_joint_angles[2])
    
    j4 = -j2 - j3
    print(f"Computed Joint 4 angle: {j4:.2f} degrees")

    # forward Kinematics
    dh_params = fk.dh_par(j1, j2, j3)
    T_matrices = fk.dh_kine(dh_params)
    ee_positions = fk.el_xyzpos(T_matrices)

    positions = np.hstack((np.array([[0], [0], [0]]), ee_positions))   # add base point (P0)

    print("\nEnd Effector Position (X, Y, Z):")
    print(ee_positions[:, -1])

    current_joint_angles = [j1, j2, j3]
    current_position = ee_positions[:, -1].tolist()  # storing the latest position of end-effector

    viz.update_plot((positions[0, :], positions[1, :], positions[2, :]))

def run_ik(viz):
    """Perform inverse kinematics and display the final result."""
    global current_joint_angles, current_position

    print("\n--- Inverse Kinematics ---")
    xt = float(input(f"Enter Target X position [current: {current_position[0]}]: ") or current_position[0])
    yt = float(input(f"Enter Target Y position [current: {current_position[1]}]: ") or current_position[1])
    zt = float(input(f"Enter Target Z position [current: {current_position[2]}]: ") or current_position[2])

    j1, j2, j3 = current_joint_angles
    j4 = -j2 - j3

    print("\nStarting IK computation...")
    damping_factor = 0.1  # to avoid large oscillations
    max_angle_step = 2.0  # max change in joint angle per iteration
    max_iterations = 500  # limit the number of iterations
    iteration = 0

    while iteration < max_iterations: 
      
        dh_params = fk.dh_par(j1, j2, j3)
        T_matrices = fk.dh_kine(dh_params)
        ee_positions = fk.el_xyzpos(T_matrices) # position of angles

  
        positions = np.hstack((np.array([[0], [0], [0]]), ee_positions))
        current_pos = ee_positions[:, -1] 

        print(f"Iteration {iteration + 1}: Current Position: {current_pos}")

        delta = np.array([[xt], [yt], [zt]]) - current_pos.reshape(3, 1)  # calculate distance to target
        distance = np.linalg.norm(delta)

        if distance < 0.01: # if target is reached, stop
            print("\nTarget reached!")
            break

        
        dXYZ = delta * damping_factor # damped motion control (smooth motion)

       
        Jac = fk.jacobian(T_matrices)  
        if np.linalg.cond(Jac) > 1e12:  # to avoid instability
            print("Jacobian is near-singular, stopping IK.")
            break

        Jac_Inv = fk.PinvJac(Jac) # preudoinverse Jacobian
        dTheta = Jac_Inv @ dXYZ # update joint angles

        dTheta = np.clip(dTheta, np.deg2rad(-max_angle_step), np.deg2rad(max_angle_step)) # +-2 degree restriction

        j1 += np.rad2deg(dTheta[0, 0])
        j2 += np.rad2deg(dTheta[1, 0])
        j3 += np.rad2deg(dTheta[2, 0])
        j4 = -j2 - j3

        print(f"Updated Joint Angles: J1={j1:.2f}, J2={j2:.2f}, J3={j3:.2f}, J4={j4:.2f}")
        iteration += 1

    if iteration == max_iterations:
        print("\nReached maximum iterations. IK did not converge.")

    current_joint_angles = [j1, j2, j3]
    current_position = current_pos.tolist()

    print("\nFinal Joint Angles:")
    print(f"J1={j1:.2f}, J2={j2:.2f}, J3={j3:.2f}, J4={j4:.2f}")

    viz.update_plot((positions[0, :], positions[1, :], positions[2, :]))

def display_menu():
    """Display the menu for the user."""
    print("\n===== Robotic Arm Kinematics =====")
    print("1. Run Forward Kinematics (FK)")
    print("2. Run Inverse Kinematics (IK)")
    print("3. Exit")
    return input("Enter your choice: ")

def main():
    """Main function to run the program."""
    viz = PlotlyKinematicsVisualization()  
    while True:
        choice = display_menu()
        if choice == '1':
            run_fk(viz)
        elif choice == '2':
            run_ik(viz)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
