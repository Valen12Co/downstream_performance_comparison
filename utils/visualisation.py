import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def visualize_3d(keypoints,keypoints2, name="3d"):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    keypoints = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
    keypoints2 = keypoints2.cpu().numpy() if torch.is_tensor(keypoints2) else keypoints2
    if keypoints.shape[0] != 17 :
        keypoints = np.insert(keypoints , 0 , values= [0,0,0], axis=0 )
        keypoints2 = np.insert(keypoints2 , 0 , values= [0,0,0], axis=0 )

    plt.figure()
    ax = plt.axes(projection='3d')

    xdata = keypoints.T[0]
    ydata = keypoints.T[1]
    zdata = keypoints.T[2]
    
    ax.scatter(xdata,ydata,zdata,color = "turquoise",label="gt")
    for i in range(17):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]] , color = "darkturquoise" )

    xdata2 = keypoints2.T[0]
    ydata2 = keypoints2.T[1]
    zdata2 = keypoints2.T[2]
    ax.scatter(xdata2,ydata2,zdata2, color ="mediumvioletred" , label ="pred")
    for i in range(17):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], zdata2[sk_points[i]] , color = "palevioletred" )

    plt.legend(loc = 'upper left' )

    #ax.axes.set_xlim3d(left=min(min(xdata),min(xdata2)), right=max(max(xdata),max(xdata2))) 
    #ax.axes.set_ylim3d(bottom=min(min(ydata),min(ydata2)), top=max(max(ydata),max(ydata2))) 
    #ax.axes.set_zlim3d(bottom=min(min(zdata),min(zdata2)), top=max(max(zdata),max(zdata2))) 
    
    ax.axes.set_xlim3d(left=-1, right = 1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 
    
    ax.grid(False)
    #ax.axes.set_xticks([-1000,0,1000])
    #ax.axes.set_yticks([-1000,0,1000])
    #ax.axes.set_zticks([-1000,0,1000])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=120, azim=60)
    
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisation', name + '.png')
    plt.savefig(name)
    plt.close()


def compute_length(keypoints):
    keypoints = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    xdata = keypoints.T[0]
    ydata = keypoints.T[1]
    zdata = keypoints.T[2]

    new_length = np.zeros([1,16])
    for i in range(16):
        diff_x = xdata[sk_points[i][0]]-xdata[sk_points[i][1]]
        diff_y = ydata[sk_points[i][0]]-ydata[sk_points[i][1]]
        if keypoints.shape[-1]==3:
            diff_z = zdata[sk_points[i][0]]-zdata[sk_points[i][1]]
            total_length = np.sqrt(diff_x**2+diff_y**2+diff_z**2)
        else:
            total_length = np.sqrt(diff_x**2+diff_y**2)

        new_length[0,i] = total_length
    return new_length

def compute_length_2D(keypoints):
    keypoints = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints

    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    xdata = keypoints.T[0]
    ydata = keypoints.T[1]

    new_length = np.zeros([keypoints.shape[0],16])
    for i in range(16):
        diff_x = xdata[sk_points[i][0]]-xdata[sk_points[i][1]]
        diff_y = ydata[sk_points[i][0]]-ydata[sk_points[i][1]]
        total_length = np.sqrt(diff_x**2+diff_y**2)

        new_length[:,i] = total_length.mean(axis=0).T#.squeeze(1)

    return new_length

def visualize_2d(keypoints,keypoints2, name="3d"):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    keypoints = keypoints.detach().cpu().numpy() if torch.is_tensor(keypoints) else keypoints
    keypoints2 = keypoints2.detach().cpu().numpy() if torch.is_tensor(keypoints2) else keypoints2
    if keypoints.shape[0] != 17 :
        keypoints = np.insert(keypoints , 0 , values= [0,0], axis=0 )
        keypoints2 = np.insert(keypoints2 , 0 , values= [0,0], axis=0 )

    plt.figure()
    fig, ax = plt.subplots()

    xdata = keypoints.T[0]
    ydata = keypoints.T[1]
    
    ax.scatter(xdata,ydata,color = "turquoise",label="gt")
    for i in range(17):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], color = "darkturquoise" )

    xdata2 = keypoints2.T[0]
    ydata2 = keypoints2.T[1]
    ax.scatter(xdata2,ydata2, color ="mediumvioletred" , label ="pred")
    for i in range(17):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color = "palevioletred" )

    plt.legend(loc = 'upper left' )

    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    ax.grid(False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisation', name + '.png')
    plt.savefig(name)
    plt.close(fig)

def visualize_on_image(ground_truth_keypoints,vitpose_keypoints, hourglass_keypoints, image, name):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    ground_truth_keypoints = ground_truth_keypoints.detach().cpu().numpy() if torch.is_tensor(ground_truth_keypoints) else ground_truth_keypoints
    vitpose_keypoints = vitpose_keypoints.detach().cpu().numpy() if torch.is_tensor(vitpose_keypoints) else vitpose_keypoints
    hourglass_keypoints = hourglass_keypoints.detach().cpu().numpy() if torch.is_tensor(hourglass_keypoints) else hourglass_keypoints
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else image

    if ground_truth_keypoints.shape[0] != 17 :
        ground_truth_keypoints = np.insert(ground_truth_keypoints , 0 , values= [0,0], axis=0 )
        vitpose_keypoints = np.insert(vitpose_keypoints , 0 , values= [0,0], axis=0 )
        hourglass_keypoints = np.insert(hourglass_keypoints , 0 , values= [0,0], axis=0 )

    height, width, _ = image.shape
    ground_truth_keypoints = ground_truth_keypoints * np.array([width, height])
    vitpose_keypoints = vitpose_keypoints * np.array([width, height])
    hourglass_keypoints = hourglass_keypoints * np.array([width, height])

    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(image)

    xdata = ground_truth_keypoints[:, 0]
    ydata = ground_truth_keypoints[:, 1]
    ax.scatter(xdata, ydata, color="turquoise", label="gt")
    for i in range(len(sk_points)):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], color="darkturquoise")
    
    # Plot predicted keypoints and skeleton
    xdata2 = vitpose_keypoints[:, 0]
    ydata2 = vitpose_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="mediumvioletred", label="ViTPose")
    for i in range(len(sk_points)):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color="palevioletred")

    xdata2 = hourglass_keypoints[:, 0]
    ydata2 = hourglass_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="gold", label="Stacked Hourglass")
    for i in range(len(sk_points)):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color="gold")

    plt.legend(loc='upper left')

    # Set limits to match the frame dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image coordinate system
    
    ax.axis('off')  # Hide axes for better visualization
    
    ax.grid(False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisation2D', name + '.png')
    plt.savefig(name)
    plt.close(fig)

def visualize_on_image_prob_pose(ground_truth_keypoints,list_prob_pose, image, name):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    ground_truth_keypoints = ground_truth_keypoints.detach().cpu().numpy() if torch.is_tensor(ground_truth_keypoints) else ground_truth_keypoints
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else image

    if ground_truth_keypoints.shape[0] != 17 :
        ground_truth_keypoints = np.insert(ground_truth_keypoints , 0 , values= [0,0], axis=0 )
        vitpose_keypoints = np.insert(vitpose_keypoints , 0 , values= [0,0], axis=0 )
        hourglass_keypoints = np.insert(hourglass_keypoints , 0 , values= [0,0], axis=0 )

    height, width, _ = image.shape
    ground_truth_keypoints = ground_truth_keypoints * np.array([width, height])
    for i in range(len(list_prob_pose)):
        list_prob_pose[i] = list_prob_pose[i]*np.array([width, height])

    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(image)

    xdata = ground_truth_keypoints[:, 0]
    ydata = ground_truth_keypoints[:, 1]
    ax.scatter(xdata, ydata, color="turquoise", label="gt")
    for i in range(len(sk_points)):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], color="darkturquoise")
    
    # Plot predicted keypoints and skeleton
    for i in range(len(list_prob_pose)):
        xdata2 = list_prob_pose[i][:,0]
        ydata2 = list_prob_pose[i][:,1]
        ax.scatter(xdata2, ydata2, color="mediumvioletred", label="Probabilistic")
        for j in range(len(sk_points)):
            ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color="palevioletred")

    plt.legend(loc='upper left')

    # Set limits to match the frame dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image coordinate system
    
    ax.axis('off')  # Hide axes for better visualization
    
    ax.grid(False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisationprobpose', name + '.png')
    plt.savefig(name)
    plt.close(fig)

def visualize_on_COCO(vitpose_keypoints, hourglass_keypoints, image, name):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    vitpose_keypoints = vitpose_keypoints.detach().cpu().numpy() if torch.is_tensor(vitpose_keypoints) else vitpose_keypoints
    hourglass_keypoints = hourglass_keypoints.detach().cpu().numpy() if torch.is_tensor(hourglass_keypoints) else hourglass_keypoints
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else image

    if vitpose_keypoints.shape[0] != 17 :
        vitpose_keypoints = np.insert(vitpose_keypoints , 0 , values= [0,0], axis=0 )
        hourglass_keypoints = np.insert(hourglass_keypoints , 0 , values= [0,0], axis=0 )

    height, width, _ = image.shape
    vitpose_keypoints = vitpose_keypoints * np.array([width, height])
    hourglass_keypoints = hourglass_keypoints * np.array([width, height])

    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(image)

    
    # Plot predicted keypoints and skeleton
    xdata2 = vitpose_keypoints[:, 0]
    ydata2 = vitpose_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="mediumvioletred", label="ViTPose")
    for i in range(len(sk_points)):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color="palevioletred")

    xdata2 = hourglass_keypoints[:, 0]
    ydata2 = hourglass_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="gold", label="Stacked Hourglass")
    for i in range(len(sk_points)):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color="gold")

    plt.legend(loc='upper left')

    # Set limits to match the frame dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image coordinate system
    
    ax.axis('off')  # Hide axes for better visualization
    
    ax.grid(False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisation2D_COCO', name + '.png')
    plt.savefig(name)
    plt.close(fig)

def visualize_3D_all(ground_truth_keypoints,vitpose_keypoints, hourglass_keypoints, name):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    ground_truth_keypoints = ground_truth_keypoints.detach().cpu().numpy() if torch.is_tensor(ground_truth_keypoints) else ground_truth_keypoints
    vitpose_keypoints = vitpose_keypoints.detach().cpu().numpy() if torch.is_tensor(vitpose_keypoints) else vitpose_keypoints
    hourglass_keypoints = hourglass_keypoints.detach().cpu().numpy() if torch.is_tensor(hourglass_keypoints) else hourglass_keypoints

    if ground_truth_keypoints.shape[0] != 17 :
        ground_truth_keypoints = np.insert(ground_truth_keypoints , 0 , values= [0,0], axis=0 )
        vitpose_keypoints = np.insert(vitpose_keypoints , 0 , values= [0,0], axis=0 )
        hourglass_keypoints = np.insert(hourglass_keypoints , 0 , values= [0,0], axis=0 )

    plt.figure()
    ax = plt.axes(projection='3d')

    xdata = ground_truth_keypoints[:, 0]
    ydata = ground_truth_keypoints[:, 1]
    zdata = ground_truth_keypoints[:, 2]
    ax.scatter(xdata, ydata, color="turquoise", label="gt")
    for i in range(len(sk_points)):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]], color="darkturquoise")
    
    # Plot predicted keypoints and skeleton
    xdata2 = vitpose_keypoints[:, 0]
    ydata2 = vitpose_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="mediumvioletred", label="ViTPose")
    for i in range(len(sk_points)):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]], color="palevioletred")

    xdata2 = hourglass_keypoints[:, 0]
    ydata2 = hourglass_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="gold", label="Stacked Hourglass")
    for i in range(len(sk_points)):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]], color="gold")


    plt.legend(loc='upper left')

    # Set limits to match the frame dimensions
    ax.axes.set_xlim3d(left=-1, right = 1) 
    ax.axes.set_ylim3d(bottom=-1, top= 1) 
    ax.axes.set_zlim3d(bottom=-1, top= 1) 
    
    ax.axis('off')  # Hide axes for better visualization
    
    ax.grid(False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=120, azim=60)
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisation3D', name + '.png')
    plt.savefig(name)
    plt.close()

def visualize_coco_image(ground_truth_keypoints,coco_keypoints, image, name):
    #Keypoints = gt, keypoints_2 = predicted
    #breakpoint()
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    ground_truth_keypoints = ground_truth_keypoints.detach().cpu().numpy() if torch.is_tensor(ground_truth_keypoints) else ground_truth_keypoints
    coco_keypoints = coco_keypoints.detach().cpu().numpy() if torch.is_tensor(coco_keypoints) else coco_keypoints
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else image

    if ground_truth_keypoints.shape[0] != 17 :
        ground_truth_keypoints = np.insert(ground_truth_keypoints , 0 , values= [0,0], axis=0 )
        coco_keypoints = np.insert(coco_keypoints , 0 , values= [0,0], axis=0 )

    height, width, _ = image.shape
    ground_truth_keypoints = ground_truth_keypoints * np.array([width, height])
    coco_keypoints = coco_keypoints * np.array([width, height])

    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(image)

    xdata = ground_truth_keypoints[:, 0]
    ydata = ground_truth_keypoints[:, 1]
    ax.scatter(xdata, ydata, color="turquoise", label="gt")
    for i in range(len(sk_points)):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], color="darkturquoise")

    xdata2 = coco_keypoints[:, 0]
    ydata2 = coco_keypoints[:, 1]
    ax.scatter(xdata2, ydata2, color="gold", label="COCO")
    for i in range(len(sk_points)):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], color="gold")

    plt.legend(loc='upper left')

    # Set limits to match the frame dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image coordinate system
    
    ax.axis('off')  # Hide axes for better visualization
    
    ax.grid(False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    name = os.path.join('/mnt/vita/scratch/vita-students/users/perret/downstream_performance_comparison/visualisation2DCOCO', name + '.png')
    plt.savefig(name)
    plt.close(fig)