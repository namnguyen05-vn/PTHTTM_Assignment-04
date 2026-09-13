"""Recompute published metrics and produce tables/figures from real saved outputs."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT,DATASETS,load_data,data_root
from .train import classification_metrics

FIG=ROOT/'figures'
COLORS={'numpy':'#246a98','pytorch':'#bc4c30','tensorflow':'#377e60'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110,'savefig.dpi':160})

def save(fig,name):
    fig.savefig(FIG/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def audit_data():
    audits=[];sources=[];FIG.mkdir(exist_ok=True)
    for name,cfg in DATASETS.items():
        d=load_data(name);x,y=d['x'],d['y'];ti,vi=d['train_ids'],d['val_ids'];xt=d['x_test']
        def hashes(a):return [hashlib.sha256(im.tobytes()).digest() for im in a]
        h=hashes(x);ht=hashes(xt);train_hashes={h[i] for i in ti};val_hashes={h[i] for i in vi}
        test_set=set(ht)
        audit=dict(dataset=name,dtype=str(x.dtype),pixel_min=int(x.min()),pixel_max=int(x.max()),finite=bool(np.isfinite(x).all()),labels_valid=bool(np.all((y>=0)&(y<cfg['classes']))),split_indices_disjoint=bool(set(ti).isdisjoint(set(vi))),all_training_indices_used=bool(len(ti)+len(vi)==len(y)),unique_training_images=len(set(h)),exact_image_hashes_shared_train_validation=len(train_hashes&val_hashes),exact_image_hashes_shared_train_test=len(train_hashes&test_set),exact_image_hashes_shared_validation_test=len(val_hashes&test_set),duplicate_policy='Keep official benchmark split; report exact pixel duplicates as a limitation. No post-test filtering.')
        audits.append(audit)
        source=data_root()/(name+'_source.json')
        if source.exists():sources.append(json.loads(source.read_text()))
        classes=list(range(10)) if cfg['classes']==10 else list(range(0,100,10))
        fig,axes=plt.subplots(2,10,figsize=(11,3.1))
        for col,c in enumerate(classes):
            ids=ti[y[ti]==c][:2]
            for row,ix in enumerate(ids):
                ax=axes[row,col];im=x[ix].transpose(1,2,0)
                ax.imshow(im[:,:,0] if cfg['channels']==1 else im,cmap='gray' if cfg['channels']==1 else None,vmin=0,vmax=255)
                ax.axis('off')
                if row==0:ax.set_title(str(d['class_names'][c]).replace('_','\n'),fontsize=8)
        fig.suptitle(name.upper()+' - examples from the training split',fontsize=13)
        fig.tight_layout();save(fig,f'{name}_samples.png')
        fig,ax=plt.subplots(figsize=(9,2.6));counts=np.bincount(y[ti],minlength=cfg['classes'])
        ax.bar(np.arange(cfg['classes']),counts,color='#246a98');ax.set(xlabel='Class index',ylabel='Training images',title=name.upper()+' - class distribution')
        fig.tight_layout();save(fig,f'{name}_distribution.png')
    (ROOT/'results'/'dataset_audit.json').write_text(json.dumps(audits,indent=2))
    (ROOT/'results'/'data_sources.json').write_text(json.dumps(sources,indent=2))
    return audits

def analyze(require_complete=True):
    FIG.mkdir(exist_ok=True);records=[];checks=[]
    for name,cfg in DATASETS.items():
        for backend in COLORS:
            for variant in ['baseline','improved']:
                folder=ROOT/'results'/f'{name}_{backend}_{variant}'
                if not (folder/'metrics.json').exists():
                    if require_complete:raise FileNotFoundError(f'Unfinished experiment: {folder.name}')
                    continue
                m=json.loads((folder/'metrics.json').read_text())
                with np.load(folder/'test_outputs.npz') as a:
                    actual,cm=classification_metrics(a['labels'],a['logits'],cfg['classes'])
                for key,value in actual.items():assert np.isclose(value,m[key],atol=1e-7), (folder,key)
                assert np.array_equal(cm,np.loadtxt(folder/'confusion_matrix.csv',delimiter=',',dtype=int))
                history=pd.read_csv(folder/'history.csv');assert len(history)==m['epochs']
                assert int(history.loc[history.val_loss.idxmin(),'epoch'])==m['best_epoch']
                records.append(m);checks.append(dict(experiment=folder.name,metrics_recomputed=True,confusion_verified=True,best_epoch_verified=True))
    df=pd.DataFrame(records);df.to_csv(ROOT/'results'/'summary.csv',index=False)
    (ROOT/'results'/'output_verification.json').write_text(json.dumps(checks,indent=2))
    for name,cfg in DATASETS.items():
        subset=df[df.dataset==name]
        if len(subset)!=6:continue
        fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
        for col,backend in enumerate(COLORS):
            for variant,style in [('baseline','--'),('improved','-')]:
                h=pd.read_csv(ROOT/'results'/f'{name}_{backend}_{variant}'/'history.csv')
                axes[0,col].plot(h.epoch,h.train_loss,style,color='#8694a0',label=variant+' train')
                axes[0,col].plot(h.epoch,h.val_loss,style,color=COLORS[backend],label=variant+' validation')
                axes[1,col].plot(h.epoch,100*h.train_accuracy,style,color='#8694a0')
                axes[1,col].plot(h.epoch,100*h.val_accuracy,style,color=COLORS[backend])
            axes[0,col].set_title(backend);axes[0,col].set_ylabel('Cross-entropy');axes[1,col].set(ylabel='Accuracy (%)',xlabel='Epoch')
            axes[0,col].legend(fontsize=7);axes[0,col].grid(alpha=.2);axes[1,col].grid(alpha=.2)
        fig.suptitle(name.upper()+' - training and validation');fig.tight_layout();save(fig,f'{name}_curves.png')
        # Choose representative model using validation loss, never test accuracy.
        chosen=subset.loc[subset.best_val_loss.idxmin()];folder=ROOT/'results'/f'{name}_{chosen.backend}_{chosen.variant}'
        d=load_data(name);names=d['class_names'];cm=np.loadtxt(folder/'confusion_matrix.csv',delimiter=',',dtype=int)
        fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(cm/cm.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
        if cfg['classes']==10:
            ax.set_xticks(range(10),names,rotation=45,ha='right');ax.set_yticks(range(10),names)
            for i in range(10):
                for j in range(10):
                    value=cm[i,j]/cm[i].sum()
                    ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=7,color='white' if value>.5 else '#183240')
        ax.set(xlabel='Predicted class',ylabel='True class',title=f'{name.upper()} - {chosen.backend} / {chosen.variant}')
        fig.colorbar(im,ax=ax,label='Fraction within true class');fig.tight_layout();save(fig,f'{name}_confusion.png')
        preds=pd.read_csv(folder/'predictions.csv');errors=preds[preds.true_label!=preds.predicted_label].sort_values('confidence',ascending=False)
        fig,axes=plt.subplots(2,5,figsize=(10,4.6))
        for ax,(_,row) in zip(axes.flat,errors.head(10).iterrows()):
            im=d['x_test'][int(row.test_id)].transpose(1,2,0)
            ax.imshow(im[:,:,0] if cfg['channels']==1 else im,cmap='gray' if cfg['channels']==1 else None,vmin=0,vmax=255)
            ax.set_title(f'T: {names[int(row.true_label)]}\nP: {names[int(row.predicted_label)]} ({row.confidence:.2f})',fontsize=8);ax.axis('off')
        fig.suptitle(name.upper()+' - ten most confident wrong predictions');fig.tight_layout();save(fig,f'{name}_errors.png')
        pairs=cm.copy();np.fill_diagonal(pairs,0);top=np.dstack(np.unravel_index(np.argsort(pairs.ravel())[-10:][::-1],pairs.shape))[0]
        detail=dict(dataset=name,representative_backend=chosen.backend,representative_variant=chosen.variant,selection='lowest best validation loss among six runs',worst_pairs=[dict(true=str(names[i]),predicted=str(names[j]),count=int(cm[i,j])) for i,j in top])
        (ROOT/'results'/f'{name}_error_analysis.json').write_text(json.dumps(detail,indent=2))
        recalls=np.diag(cm)/cm.sum(1);order=np.argsort(recalls)[:min(20,len(names))]
        fig,ax=plt.subplots(figsize=(8,4));ax.barh(np.arange(len(order)),100*recalls[order],color='#246a98');ax.set_yticks(range(len(order)),names[order]);ax.set(xlabel='Recall (%)',xlim=(0,100),title=name.upper()+' - lowest class recalls');ax.invert_yaxis();fig.tight_layout();save(fig,f'{name}_recall.png')
    if len(df)==18:
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,name in zip(axes,DATASETS):
            for j,v in enumerate(['baseline','improved']):
                vals=[float(df[(df.dataset==name)&(df.backend==b)&(df.variant==v)].accuracy.iloc[0])*100 for b in COLORS]
                bars=ax.bar(np.arange(3)+(j-.5)*.34,vals,.34,label=v,color=['#87aec5','#214f70'][j])
                ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=3)
            ax.set_xticks(range(3),list(COLORS),rotation=15);ax.set(ylim=(0,110),ylabel='Test accuracy (%)',title=name.upper());ax.legend(fontsize=8)
        fig.tight_layout();save(fig,'accuracy_comparison.png')
    print(df[['dataset','backend','variant','accuracy','macro_f1','best_epoch']].to_string(index=False))
    return df

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    if a.audit_only:print(json.dumps(audit_data(),indent=2))
    else:analyze(not a.allow_partial)
