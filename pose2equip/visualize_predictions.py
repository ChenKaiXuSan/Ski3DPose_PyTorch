#!/usr/bin/env python3
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn; import torch

class _MD(nn.Module):
    def __init__(s, n=None):
        super().__init__()
        s.config = type('C',(),{'hidden_size':256})(); s.proj = nn.Linear(256,256)
    @classmethod
    def from_pretrained(c,n): return c(n)
    def forward(s, pixel_values):
        B_,_,H_,W_ = pixel_values.shape; N=247; d=pixel_values.device
        class O: pass
        o = O(); o.last_hidden_state = torch.cat([torch.zeros(B_,1,256,device=d), torch.randn(B_,N,256,device=d)], dim=1); return o

_tf = types.ModuleType('transformers'); _tf.AutoModel = _MD; sys.modules['transformers'] = _tf
from pose2equip.models.pose2equip_net import Pose2EquipNetImproved
from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS
import numpy as np; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visual_output')
os.makedirs(OUT, exist_ok=True)
BONES = [(14,2),(2,4),(4,13),(14,3),(3,5),(5,12),(14,6),(14,7),(6,8),(8,10),(7,9),(9,11)]
EQ = ['ski_L','ski_R','pole_L','pole_R']
EC = {'ski_L':'#e74c3c','ski_R':'#e67e22','pole_L':'#3498db','pole_R':'#9b59b6'}

def jbones(ax,J,co='gray',al=0.6,s=40):
    for i in range(len(J)): ax.scatter(*J[i],c=co,s=s,edgecolors='white',lw=0.5,alpha=al*2,zorder=4)
    for a,b in BONES: ax.plot([J[a,0],J[b,0]],[J[a,1],J[b,1]],[J[a,2],J[b,2]],co,lw=1.5,alpha=al)

def sv(ax):
    ax.set_xlim([-2.5,2.5]); ax.set_ylim([-2.5,2.5]); ax.set_zlim([-2.5,2.5])
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

def pv(pred,gt):
    fig,axes = plt.subplots(2,2,figsize=(10,10),subplot_kw={'projection':'3d'})
    for i,eq in enumerate(EQ):
        ax=axes[i//2][i%2]; p,g=pred[i],gt[i]; c=EC[eq]
        ax.plot([p[0,0],p[1,0]],[p[0,1],p[1,1]],[p[0,2],p[1,2]],c,lw=3.5,alpha=0.9,label='Pred',zorder=5)
        for e in range(2): ax.scatter(*p[e],c=c,s=120,edgecolors='white',lw=1.5,zorder=6)
        ax.plot([g[0,0],g[1,0]],[g[0,1],g[1,1]],[g[0,2],g[1,2]],c,lw=2.5,linestyle='--',alpha=0.6,label='GT',zorder=4)
        sv(ax); ax.view_init(elev=20,azim=-50); t='Ski' if 'ski' in eq else 'Pole'; ax.set_title(f'{t} ({eq})')
    axes[1][1].legend(loc='upper right',fontsize=9)
    fig.suptitle('Pose2EquipNetImproved - Prediction vs Ground Truth',fontsize=13,y=0.98)
    plt.tight_layout(); p=os.path.join(OUT,'pred_vs_gt_3d.png'); plt.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()

def ps(pred,gt,joints):
    fig=plt.figure(figsize=(14,7))
    for idx,(obj,title) in enumerate([(pred,'Pose2EquipNetImproved - Predicted'),(gt,'Ground Truth')],1):
        ax=fig.add_subplot(1,2,idx,projection='3d'); ig=(idx==2); jbones(ax,joints,'gray',0.6,50)
        for i,eq in enumerate(EQ):
            pts=obj[i]; c=EC[eq]
            ax.plot([pts[0,0],pts[1,0]],[pts[0,1],pts[1,1]],[pts[0,2],pts[1,2]],c,lw=4,alpha=0.95,linestyle='--' if ig else None,label=f'{chr(71)+chr(84) if ig else "pred"} {eq}')
            for e in range(2): ax.scatter(*pts[e],c=c if not ig else 'none',edgecolors=c,s=120,lw=2,zorder=6)
        sv(ax); ax.view_init(elev=25,azim=-60); ax.set_title(title); ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Full Scene (Pred|GT)',fontsize=13); plt.tight_layout()
    p=os.path.join(OUT,'scene_3d.png'); plt.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()

def pe(pred,gt):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    errs=np.array([np.linalg.norm(pred[q]-gt[q],axis=1) for q in range(4)]); cols=[EC[eq] for eq in EQ]; x=np.arange(4)
    ax1.bar(x,errs[:,0],width=0.4,label='ep1',color=cols,alpha=0.8)
    ax1.bar(x,errs[:,1],bottom=errs[:,0],width=0.4,label='ep2',color=cols,alpha=0.5)
    tot=np.sum(errs,axis=1); bars=ax2.bar(range(4),tot,width=0.6,color=cols)
    for b,v in zip(bars,tot): ax2.text(b.get_x()+b.get_width()/2,v+0.005,f'{v:.3f}m',ha='center',fontsize=10,fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(EQ); ax1.legend(fontsize=9)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(EQ)
    fig.suptitle('Error Analysis',fontsize=13); plt.tight_layout()
    p=os.path.join(OUT,'error_analysis.png'); plt.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()

np.random.seed(42); torch.manual_seed(42)
model = Pose2EquipNetImproved(num_joints=15,hidden_dim=256,dino_freeze=True,target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,decoder_layers=3,num_heads=8)
dev='cuda' if torch.cuda.is_available() else 'cpu'
enc = DinoPatchEncoder()
if dev == 'cuda': enc = enc.cuda()
model.image_encoder = enc
np.random.seed(42)
joints=np.random.randn(15,3)*1.5; joints[:,2]*=0.5
gt=np.array([[-0.8,-0.3,-0.9],[0.8,0.3,-0.7],[-0.8,0.3,-0.9],[0.8,-0.3,-0.7],[0.2,1.5,0.4],[0.2,0.3,0.0],[-0.2,1.5,0.4],[-0.2,0.3,0.0]])*0.8
frame=torch.zeros(1,3,224,224); pose=torch.from_numpy(joints).float()
with torch.no_grad(): out=model(human_frame=frame.unsqueeze(1),human_3d=pose.unsqueeze(0).unsqueeze(1))
pred=np.clip(out['object_3d'][0].cpu().numpy(),-3,3); gtc=np.clip(gt,-3,3)
pv(pred,gtc); ps(pred,gtc,joints); pe(pred,gtc)
print(f'All plots saved to {OUT}')
