"""Python figure contract: random test image plate + faithful categorical evidence.

Claim: uniformly sampled test tiles reveal target/state agreement and joint errors.
Panels: model-input pre/post RGB; GT/pred target; GT/pred foreground state;
joint-error map; spatial error overlay. No content/result selection or contrast edits.
Input images256; scored maps64 displayed nearest (no extra detail invented).
Diagnostic export: PNG300dpi plus SVG/PDF with editable text, not a journal submission.
Source data: saved NPZ and deterministic sampling manifest; no fabricated values.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

TARGETS=['Farmland','Runway','Taxiway','Apron','Bridge','Highway','Building','Shelter',
         'Tower','Pier','Dock','Tank','Aircraft','Vessel','Crater','Background']
TARGET_COLORS=['#8BA17A','#6B85B5','#63A4AD','#B29AC5','#9E8780','#D5AC61','#C97878','#C29B83',
               '#9294AE','#86A29D','#A39281','#A2A272','#D88753','#5A718A','#B78F9A','#E0E2E4']
STATES=['no-change','damaged','reduced','added','extended','replaced','none / background']
STATE_COLORS=['#647687','#B65F60','#967CAB','#479F96','#C4A15A','#587DBC','#E0E2E4']
ERROR_NAMES=['Correct background','Correct foreground','Missed foreground','Wrong target / background FP','Wrong state','Ignored GT']
ERROR_COLORS=['#E0E2E4','#3E8C83','#8D67A5','#D6873A','#4576AA','#FFFFFF']


def rgb_map(values, colors):
    from matplotlib.colors import to_rgb
    lut=np.ones((256,3))
    for i,color in enumerate(colors):lut[i]=to_rgb(color)
    return lut[values.astype(int)]


def error_map(gt,gs,pt,ps):
    valid=(gt!=255)&(gs!=255)
    result=np.zeros(gt.shape,dtype=np.uint8)
    result[valid&(gt!=15)&(pt==gt)&(ps==gs)]=1
    result[valid&(gt!=15)&(pt==15)]=2
    result[valid&(pt!=gt)&(pt!=15)]=3
    result[valid&(gt!=15)&(pt==gt)&(ps!=gs)]=4
    result[~valid]=5
    return result


def ignored_hatch(ax,ignore):
    if ignore.any():
        ax.contourf(np.arange(64)+.5,np.arange(64)+.5,ignore.astype(int),
                    levels=[.5,1.5],colors='none',hatches=['////'],alpha=.2)


def panel(ax,image,title,ignore=None):
    ax.imshow(image,interpolation='nearest',extent=(0,64,64,0))
    if ignore is not None:ignored_hatch(ax,ignore)
    ax.set_title(title,fontsize=11,pad=5)
    ax.set_xlim(0,64);ax.set_ylim(64,0);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--test-dir',required=True)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--overview-only',action='store_true');opts=parser.parse_args()
    root=Path(opts.test_dir)
    if opts.preview:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from evaluation.strict_joint_protocol import StrictJointEvaluator
        details=[]
        for number,record in enumerate(json.loads((root/'sampling.json').read_text())['selected'],1):
            data_file=f'sample_{number:02d}.npz'
            if not (root/'samples'/data_file).exists():continue
            sample=np.load(root/'samples'/data_file)
            score=StrictJointEvaluator().update(sample['pred_target'],sample['pred_state'],sample['gt_target'],sample['gt_state']).compute()
            details.append({'sample_number':number,'dataset_index':record['dataset_index'],'data_file':data_file,'metrics':score})
    else:
        details=json.loads((root/'sample_details.json').read_text())
    output=root/('figures_preview' if opts.preview else 'figures');output.mkdir(exist_ok=opts.overview_only)
    error_cmap=ListedColormap(ERROR_COLORS);error_norm=BoundaryNorm(np.arange(7)-.5,6)
    for record in ([] if opts.overview_only else details):
        sample=np.load(root/'samples'/record['data_file'])
        gt,gs,pt,ps=[sample[k] for k in ('gt_target','gt_state','pred_target','pred_state')]
        ignore=(gt==255)|(gs==255)
        display_ps=ps.copy();display_ps[pt==15]=6
        errors=error_map(gt,gs,pt,ps)
        fig,axes=plt.subplots(2,4,figsize=(12.8,7.4))
        fig.subplots_adjust(left=.025,right=.975,top=.90,bottom=.22,wspace=.06,hspace=.22)
        score=record['metrics']['strict_joint_f1']
        score_label='n/a (no scored foreground)' if score is None else f'{score:.1%}'
        fig.suptitle(f'Random test sample {record["sample_number"]}  |  index {record["dataset_index"]}  |  joint F1 {score_label}',fontsize=13,y=.98)
        panel(axes[0,0],sample['img_t1'],'a  Pre image (model input)')
        panel(axes[0,1],sample['img_t2'],'b  Post image (model input)')
        panel(axes[0,2],rgb_map(gt,TARGET_COLORS),'c  GT target',ignore)
        panel(axes[0,3],rgb_map(pt,TARGET_COLORS),'d  Pred target',ignore)
        panel(axes[1,0],rgb_map(gs,STATE_COLORS),'e  GT state',ignore)
        panel(axes[1,1],rgb_map(display_ps,STATE_COLORS),'f  Pred state (foreground)',ignore)
        panel(axes[1,2],errors,'g  Joint errors')
        axes[1,2].images[0].set_cmap(error_cmap);axes[1,2].images[0].set_norm(error_norm)
        ignored_hatch(axes[1,2],ignore)
        panel(axes[1,3],sample['img_t2'],'h  Errors over post image')
        axes[1,3].imshow(np.ma.masked_where((errors<2)|(errors==5),errors),cmap=error_cmap,norm=error_norm,
                         interpolation='nearest',extent=(0,64,64,0),alpha=.7)
        ignored_hatch(axes[1,3],ignore)
        present=sorted(set(gt[gt!=255].reshape(-1).tolist()+pt.reshape(-1).tolist()))
        fig.legend([Patch(facecolor=TARGET_COLORS[i]) for i in present],[TARGETS[i] for i in present],
                   loc='lower center',bbox_to_anchor=(.5,.15),ncol=min(len(present),8),fontsize=9,frameon=False,title='Target colors')
        fig.legend([Patch(facecolor=c) for c in STATE_COLORS],STATES,loc='lower center',bbox_to_anchor=(.5,.075),
                   ncol=7,fontsize=9,frameon=False,title='State colors')
        fig.legend([Patch(facecolor=c,hatch='////' if i==5 else None) for i,c in enumerate(ERROR_COLORS)],ERROR_NAMES,
                   loc='lower center',bbox_to_anchor=(.5,.015),ncol=6,fontsize=8,frameon=False)
        fig.text(.5,.004,'Uniform random sampling, seed20261009. Scored grid64, shown nearest. Hatching = ignored GT. Predicted Farmland state fixed to no-change.',
                 ha='center',fontsize=8)
        name=f'sample_{record["sample_number"]:02d}'
        fig.savefig(output/f'{name}.png',dpi=300,facecolor='white')
        fig.savefig(output/f'{name}.pdf',facecolor='white')
        fig.savefig(output/f'{name}.svg',facecolor='white')
        plt.close(fig)
    # Compact overview: all six random samples, without choosing the best-looking ones.
    fig,axes=plt.subplots(len(details),4,figsize=(10.8,2.6*len(details)+.7),squeeze=False)
    fig.subplots_adjust(left=.055,right=.98,top=.925,bottom=.07,wspace=.04,hspace=.12)
    for row,record in enumerate(details):
        sample=np.load(root/'samples'/record['data_file']);gt,gs,pt,ps=[sample[k] for k in ('gt_target','gt_state','pred_target','pred_state')]
        ignore=(gt==255)|(gs==255);errors=error_map(gt,gs,pt,ps)
        for col,(image,title) in enumerate(((sample['img_t2'],'Post image'),(rgb_map(gt,TARGET_COLORS),'GT target'),
                                            (rgb_map(pt,TARGET_COLORS),'Pred target'),(errors,'Joint errors'))):
            panel(axes[row,col],image,title if row==0 else '',ignore if col in (1,2) else None)
        axes[row,3].images[0].set_cmap(error_cmap);axes[row,3].images[0].set_norm(error_norm)
        ignored_hatch(axes[row,3],ignore)
        axes[row,0].text(-.08,.5,f'Sample {record["sample_number"]}\nindex {record["dataset_index"]}',transform=axes[row,0].transAxes,
                         ha='right',va='center',rotation=90,fontsize=9)
    fig.suptitle(f'{len(details)} uniformly random test samples | epoch50 + fixed Farmland semantics',fontsize=13,y=.985)
    fig.legend([Patch(facecolor=TARGET_COLORS[i]) for i in (0,1,2,3,5,6,12,13,15)],
               [TARGETS[i] for i in (0,1,2,3,5,6,12,13,15)],loc='lower center',bbox_to_anchor=(.5,.035),ncol=5,fontsize=9,frameon=False)
    fig.legend([Patch(facecolor=c,hatch='////' if i==5 else None) for i,c in enumerate(ERROR_COLORS)],ERROR_NAMES,
               loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,fontsize=9,frameon=False)
    fig.savefig(output/'random_test_overview.png',dpi=300,facecolor='white')
    fig.savefig(output/'random_test_overview.pdf',facecolor='white')
    fig.savefig(output/'random_test_overview.svg',facecolor='white')
    plt.close(fig)
    (output/'figure_QA.json').write_text(json.dumps({
        'contract':'Random diagnostic image plate; no performance-based example selection',
        'backend':'Python matplotlib only','n_images':len(details),'sampling_seed':20261009,
        'source_arrays':'samples/*.npz','no_crops_or_contrast_adjustments':True,
        'input_image_hw':[256,256],'scored_maps_hw':[64,64],'mask_display_interpolation':'nearest',
        'background_predicted_state_rendered_as_none':'display only; state-head diagnostics keep its actual predicted state',
        'GT_ignore':'hatching, excluded from score; pred colors retained underneath',
        'statistical_intervals':'not estimated; examples are diagnostic, not representativeness proof',
        'export':'PNG300dpi and editable-text SVG/PDF; report display, not submission-grade claim',
        'accepted_source_QA_warnings':['Report image-plate width exceeds manuscript column size by design; no TIFF requested',
                                       'PNG300dpi report export; 256px inputs/64px maps do not gain information at600dpi']},indent=2))
    print(output)


if __name__=='__main__':main()
