import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch


def visualize_3d_heatmap(heatmap , heatmap_pred = None , name="3d"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = ListedColormap(sns.color_palette("flare", 64).as_hex())
    for i in range(0,17):
        tmp = heatmap[0,i,:,:,:]
        
        tmp[tmp[:,:,:]<0.0001] = 0.0
        indices = torch.nonzero(tmp)
        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]
    
        sc = ax.scatter(x, y, z, s=10, c=tmp[x,y,z]*10, marker='o',  cmap=cmap, alpha=0.5)

    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)     
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()
    plt.savefig(name)
    plt.close()


def visualize_3d(keypoints,keypoints2, name="3d"):
    
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

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

    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1 , top=1 ) 
    
    ax.grid(False)
    ax.axes.set_xticks([-1,0,1])
    ax.axes.set_yticks([-1,0,1])
    ax.axes.set_zticks([-1,0,1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=120, azim=60)
    
    plt.savefig(name)
    # plt.savefig("./"+run_num+"/"+name +'.png')
    # plt.show()

def visualize_2d(keypoints,st_kp=None, frame=None, name = "kp"):
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    
    if keypoints.shape[0] != 17 :
        print("***** visualizations for 2d with 16 ky *****")
        keypoints = np.insert(keypoints , 0 , values= [0,0], axis=0 )
        st_kp = np.insert(st_kp , 0 , values= [(st_kp[0,0]+st_kp[5,0])/2,(st_kp[0,1]+st_kp[5,1])/2], axis=0 )

    plt.figure()

    frame = cv2.resize(frame, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
    keypoints[:,0] = keypoints[:,0] *1000
    keypoints[:,1] = keypoints[:,1] *1000
    st_kp[:,0] = st_kp[:,0] *1000
    st_kp[:,1] = st_kp[:,1] *1000
    plt.imshow(frame)


    plt.plot(keypoints.T[0],keypoints.T[1], "o", color="turquoise", markersize=3)
    for i in range(17):
        plt.plot(keypoints.T[0][sk_points[i]], keypoints.T[1][sk_points[i]], color = "darkturquoise" )
    plt.plot(st_kp.T[0],st_kp.T[1], "o", color = "mediumvioletred", markersize=3)
    for i in range(17):
        plt.plot(st_kp.T[0][sk_points[i]], st_kp.T[1][sk_points[i]], color = "palevioletred" )

    # plt.xlim([-1,1]), plt.ylim([1,-1])

    plt.savefig(name)
    
    plt.close()


def plot_heat_map(data):
    # data = np.random.rand(10, 10)

    # plot heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='hot')

    # set colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set axis labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title('Heatmap Title')

    # show plot
plt.show()

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]


camera_parameters =  {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}


# (Quaternion Calculations -> source : https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/ )
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def qv_mult(q1, v1):
    # q2 = (0.0,) + v1
    q2 = np.insert(v1,0,0) #new
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]



def plot_losses(epoch_losses,epoch_eval_loss,epoch_metric,epoch_eval_metric, run_name) :

    plt.figure(figsize=(20,6))
    plt.subplot(1, 2, 1)

    plt.plot(epoch_losses, color = "darkturquoise")
    plt.plot(epoch_eval_loss, color = "palevioletred")

    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.legend(["training","validation"])

    plt.subplot(1, 2, 2)
    plt.plot(epoch_metric, color = "darkturquoise")
    plt.plot(epoch_eval_metric, color = "palevioletred")

    plt.xlabel("epoch")
    plt.ylabel("MPJPE")

    plt.legend(["training","validation"])
    
    plt.savefig(run_name+"/plot_metric.pdf")
    plt.show()
   
   
    
import copy
def flip_pose(data):
    
    shape = data.shape #new is N,J,D for tensores and J,D in getitem
    """
    This function is from MotionBERT and modified
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    # flipped_data = copy.deepcopy(data) new
    if len(shape)==3: #it means its a tensor in trainig loop
        flipped_data =  data.clone()
    elif len(shape)==2:
        flipped_data =  data.copy()
    
    if shape[-1] == 2: #new
        flipped_data[..., 0] = 1 - flipped_data[..., 0]  # flip x of all joints
    elif shape[-1] == 3:
        flipped_data[..., 0] *= -1 # flip x of all joints
        
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data