import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])

def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30  
    azim = -45  
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        if len(pcd[0])==3:
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        else:
            cmap = plt.cm.Spectral
            norm = plt.Normalize(vmin=0, vmax=1)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, s=size,c = cmap(norm(pcd[:, 3])))
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def save_image(filename,pcds,titles):
    fig = plt.figure()
    num_plot=len(titles)
    for i in range(num_plot):
        pcd=pcds[i]
        ax = fig.add_subplot(1,num_plot,i+1,projection='3d')
        if i ==0:
            valid_points=[]
            for point in pcd:
                if sum(point)!=0:
                    valid_points.append(list(point))
            pcd = np.asarray(valid_points)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=0.5,color='red')
        ax.set_title(titles[i])
        ax.set_axis_off()
        ax.set_box_aspect((np.ptp(pcd[:, 0]), np.ptp(pcd[:, 1]), np.ptp(pcd[:, 2])))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.axis('equal')
    fig.savefig(filename)
    plt.close(fig)

def save_overlap_image(filename,pcds,titles):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    num_plot=len(titles)
    colors=['red','purple','gray']
    s_list=[6,3,0.2]
    for i in range(num_plot):
        pcd=pcds[i]
        if i ==0:
            valid_points=[]
            for point in pcd:
                if sum(point)!=0:
                    valid_points.append(list(point))
            pcd = np.asarray(valid_points)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=s_list[i],color=colors[i])
        ax.set_title(titles[i])
        ax.set_axis_off()
        ax.set_box_aspect((np.ptp(pcd[:, 0]), np.ptp(pcd[:, 1]), np.ptp(pcd[:, 2])))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.axis('equal')
    filename=filename.replace('.png','_overlap.png')
    fig.savefig(f'{filename}')
    plt.close(fig)

def plot_pcd_one_view_new(filename, pcds, titles):
    save_image(filename,pcds,titles)
    save_overlap_image(filename,pcds,titles)
