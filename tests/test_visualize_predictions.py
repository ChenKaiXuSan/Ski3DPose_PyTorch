#!/usr/bin/env python3
"""Visualize Pose2quipNetImproved predictions on synthetic data.

Outputs in visual_output/:
  1. pred_vs_gt_3d.png    — per-equipment: prediction (solid) vs GT (dashed)
  2. scene_3d.png         — full human skeleton + predicted skis/poles (pred | GT side-by-side)
  3. error_analysis.png   — per-endpoint L2 errors, stacked bar chart
"""

import sys, os, types
# Add dual2quip root to sys.path so 'from dual2quip...' imports resolve
sys.path.insert(0, '/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2quip')

import torch.nn as nn; import torch

# Patch transformers AutoModel before pose2quip import
class _MockD(nn.Module):
    def __init__(s,model_name=None):
        super().__init__()
        self.register_buffer('_dev',torch.zeros(1))  # track device via .to()
        s.config=type('C',(),{'hidden_size':256})(); s.proj=nn.Linear(256,256)
    @classmethod
    def from_pretrained(c,n): return c(n)
    def forward(s,pixel_values):
        B_,_,H_,W_ = pixel_values.shape; N=247; d=pixel_values.device
        class O: pass
        o=O(); o.last_hidden_state=torch.cat([torch.zeros(B_,1,256,device=d), torch.randn(B_,N,256,device=d)],dim=1); return o
_tf=sys.modules.get('transformers',types.ModuleType('transformers'))
_tf.AutoModel=_MockD; sys.modules['transformers'] = _tf

from dual2quip.models.pose2quip_net import Pose2quipNetImproved
from dual2quip.map_config import FILTER_SKELETON_CONNECTIONS
import numpy as np; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT_DIR = '/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/tests/visual_output'
os.makedirs(OUT_DIR, exist_ok=True)

# Skeleton topology
BONES=[(14,2),(2,4),(4,13),(14,3),(3,5),(5,12),(14,6),(14,7),(6,8),(8,10),(7,9),(9,11)]
EQ  = ['ski_L','ski_R','pole_L','pole_R']
ECOL = {'ski_L':'#e74c3c','ski_R':'#e67e22','pole_L':'#3498db','pole_R':'#9b59b6'}

def _jbones(ax,J,co='gray',al=0.6,s=40):
    for i in range(len(J)):
        ax.scatter(*J[i],c=co,s=s,edgecolors='white',lw=0.5,alpha=al*2,zorder=4)
    for a,b in BONES:
        ax.plot([J[a,0],J[b,0]],[J[a,1],J[b,1]],[J[a,2],J[b,2]],co,lw=1.5,alpha=al)

def _set_view(ax):
    ax.set_xlim([-2.5,2.5]);ax.set_ylim([-2.5,2.5]);ax.set_zlim([-2.5,2.5])
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')

# ── Plot 1: pred vs GT per equipment ───────────────────────────────────
def plot_pred_vs_gt(pred, gt):
    fig, axes = plt.subplots(2,2,figsize=(10,10),subplot_kw={'projection':'3d'})
    for i, eq in enumerate(EQ):
        ax=axes[i//2][i%2]; p,g=pred[i],gt[i]; c=ECOL[eq]
        ax.plot([p[0,0],p[1,0]],[p[0,1],p[1,1]],[p[0,2],p[1,2]],c,lw=3.5,alpha=0.9,label='Pred',zorder=5)
        for e in range(2): ax.scatter(*p[e],c=c,s=120,edgecolors='white',lw=1.5,zorder=6)
        ax.plot([g[0,0],g[1,0]],[g[0,1],g[1,1]],[g[0,2],g[1,2]],c,lw=2.5,linestyle='--',alpha=0.6,label='GT',zorder=4)
        _set_view(ax); ax.view_init(elev=20,azim=-50)
        t='Ski' if 'ski' in eq else 'Pole'; ax.set_title(f'{t} ({eq})')
    axes[1][1].legend(loc='upper right',fontsize=9)
    fig.suptitle('Pose2quipNetImproved — Prediction vs Ground Truth (solid / dashed)',fontsize=13,y=0.98)
    plt.tight_layout()
    p=os.path.join(OUT_DIR,'pred_vs_gt_3d.png')
    plt.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f'  [1/3] {p}')

# ── Plot 2: scene (Pred | GT side-by-side) ─────────────────────────────
def plot_scene(pred, gt, joints):
    fig=plt.figure(figsize=(14,7))
    for idx,(obj,title) in enumerate([(pred,'Pose2quipNetImproved — Predicted'),(gt,'Ground Truth')],1):
        ax=fig.add_subplot(1,2,idx,projection='3d'); ig=(idx==2); _jbones(ax,joints,'gray',0.6,50)
        for i,eq in enumerate(EQ):
            pts=obj[i]; c=ECOL[eq]
            ls='--' if ig else None; lb=f'{chr(71)+chr(84) if ig else "pred"} {eq}'
            ax.plot([pts[0,0],pts[1,0]],[pts[0,1],pts[1,1]],[pts[0,2],pts[1,2]],c,lw=4,alpha=0.95,linestyle=ls,label=lb)
            for e in range(2): ax.scatter(*pts[e],c=c if not ig else 'none',edgecolors=c,s=120,lw=2,zorder=6)
        _set_view(ax); ax.view_init(elev=25,azim=-60)
        ax.set_title(title); ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Pose2quipNetImproved — Full Scene (Pred | GT)',fontsize=13)
    plt.tight_layout()
    p=os.path.join(OUT_DIR,'scene_3d.png')
    plt.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f'  [2/3] {p}')

# ── Plot 3: error analysis ─────────────────────────────────────────────
def plot_error(pred, gt):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    errs=np.array([np.linalg.norm(pred[q]-gt[q],axis=1) for q in range(4)])
    cols=[ECOL[eq] for eq in EQ]; x=np.arange(4)
    ax1.bar(x,errs[:,0],width=0.4,label='endpoint 1',color=cols,alpha=0.8)
    ax1.bar(x,errs[:,1],bottom=errs[:,0],width=0.4,label='endpoint 2',color=cols,alpha=0.5)
    tot=np.sum(errs,axis=1); bars=ax2.bar(range(4),tot,width=0.6,color=cols)
    for b,v in zip(bars,tot): ax2.text(b.get_x()+b.get_width()/2,v+0.005,f'{v:.3f}m',ha='center',fontsize=10,fontweight='bold')
    ax1.set_xticks(x);ax1.set_xticklabels(EQ);ax1.legend(fontsize=9)
    ax2.set_xticks(range(4));ax2.set_xticklabels(EQ)
    ax1.set_ylabel('L2 Error (m) per endpoint'); ax1.set_title('Per-Equipment Endpoint Errors')
    ax2.set_ylabel('Total L2 Error (m)'); ax2.set_title('Error Summary')
    fig.suptitle('Pose2quipNetImproved — Error Analysis',fontsize=13)
    plt.tight_layout()
    p=os.path.join(OUT_DIR,'error_analysis.png')
    plt.savefig(p,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f'  [3/3] {p}')

# ── Main ────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42); torch.manual_seed(42)

    model = Pose2quipNetImproved(num_joints=15,hidden_dim=256,dino_freeze=True,
        target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,decoder_layers=3,num_heads=8)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(dev).eval()

    print(f'\n  Model params: {sum(p.numel() for p in model.parameters()):,} | Device: {dev}')

    # Generate realistic skiing pose sample
    np.random.seed(42)
    joints=np.random.randn(15,3)*1.5; joints[:,2]*=0.5  # feet slightly lower than hands
    gt=np.array([[-0.8,-0.3,-0.9],[0.8,0.3,-0.7],     # ski_L (~1.6m)
                 [-0.8,0.3,-0.9],[0.8,-0.3,-0.7],      # ski_R (~1.6m)
                 [0.2,1.5,0.4],[0.2,0.3,0.0],          # pole_L (~1.2m)
                 [-0.2,1.5,0.4],[-0.2,0.3,0.0]])*0.8   # pole_R (~1.2m)

    # Model forward — fake frames (mock DINO ignores content)
    frame=torch.zeros(1,3,224,224).to(dev); pose=torch.from_numpy(joints).float().to(dev)
    with torch.no_grad():
        out=model(human_frame=frame.unsqueeze(1), human_3d=pose.unsqueeze(0).unsqueeze(1))
    pred=np.clip(out['object_3d'][0].cpu().numpy(),-3,3); gtc=np.clip(gt,-3,3)

    # Print equipment sizes
    for i,eq in enumerate(EQ):
        gl=np.linalg.norm(gt[i,0]-gt[i,1])
        pl=np.linalg.norm(pred[i,0]-pred[i,1])
        print(f'  {eq}: GT={gl:.3f}m Pred={pl:.3f}m')

    # Generate all plots
    print(f'\n  Generating plots -> {OUT_DIR}\n')
    plot_pred_vs_gt(pred,gtc)
    plot_scene(pred,gtc,joints)
    plot_error(pred,gtc)
    print(f'\n  All visualizations saved to: {OUT_DIR}')

if __name__ == '__main__':
    main()
