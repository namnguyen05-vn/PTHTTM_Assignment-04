"""Recompute every saved prediction; aggregate seed variation and paired ablations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT, DATASETS, load_data, read_pickle, data_root
from .train import experiment_dir, classification_metrics
from run_extended import jobs, SEEDS

OUT=ROOT/'results';FIG=ROOT/'figures'/'extended'
METRICS=['accuracy','macro_precision','macro_recall','macro_f1','top5_accuracy','test_loss']
COLORS={'numpy':'#206891','pytorch':'#bb5936','tensorflow':'#388366'}

def savefig(name):
    plt.savefig(FIG/name,dpi=190,bbox_inches='tight');plt.close()

def analyze():
    FIG.mkdir(exist_ok=True)
    records=[];checks=[];histories={};logits={}
    canonical_labels={}
    for dataset in DATASETS:
        with np.load(OUT/f'{dataset}_numpy_baseline'/'test_outputs.npz') as saved:
            canonical_labels[dataset]=saved['labels']
    for job in jobs():
        folder=experiment_dir(**job)
        cfg=json.loads((folder/'config.json').read_text())
        m=json.loads((folder/'metrics.json').read_text())
        for key in ['dataset','backend','variant','seed','ablation']:
            assert cfg.get(key)==job[key],(folder,key)
        assert cfg['epochs']==DATASETS[job['dataset']]['epochs'] and cfg['batch_size']==128
        with np.load(folder/'test_outputs.npz',allow_pickle=False) as z:
            z=dict(z);recomputed,cm=classification_metrics(z['labels'],z['logits'],DATASETS[job['dataset']]['classes'])
        np.testing.assert_array_equal(z['labels'],canonical_labels[job['dataset']])
        for k in METRICS:np.testing.assert_allclose(m[k],recomputed[k],atol=2e-6,rtol=2e-6)
        np.testing.assert_array_equal(cm,np.loadtxt(folder/'confusion_matrix.csv',delimiter=','))
        pred=pd.read_csv(folder/'predictions.csv')
        np.testing.assert_array_equal(pred.test_id,np.arange(len(z['labels'])))
        np.testing.assert_array_equal(pred.true_label,z['labels'])
        np.testing.assert_array_equal(pred.predicted_label,z['logits'].argmax(1))
        probability=np.exp(z['logits']-z['logits'].max(1,keepdims=True))
        probability/=probability.sum(1,keepdims=True)
        np.testing.assert_allclose(pred.confidence,probability.max(1),atol=1e-7,rtol=1e-6)
        h=pd.read_csv(folder/'history.csv')
        assert len(h)==cfg['epochs'] and h.epoch.tolist()==list(range(1,cfg['epochs']+1))
        assert int(h.loc[h.val_loss.idxmin(),'epoch'])==m['best_epoch']
        np.testing.assert_allclose(h.val_loss.min(),m['best_val_loss'])
        key=(job['dataset'],job['backend'],job['variant'],job['seed'],job['ablation'])
        histories[key]=h;logits[key]=z
        records.append({**job,**{k:m[k] for k in METRICS+['best_epoch','best_val_loss','parameter_count','train_seconds','validation_seconds','test_seconds']},'folder':folder.relative_to(ROOT).as_posix()})
        checks.append({'folder':folder.relative_to(ROOT).as_posix(),'metrics_predictions_confusion_best_epoch_verified':True})
    frame=pd.DataFrame(records);frame.to_csv(OUT/'extended_runs.csv',index=False)
    main=frame[frame.ablation.isna()];assert len(main)==54
    group=main.groupby(['dataset','backend','variant'])[METRICS].agg(['mean','std'])
    group.columns=['_'.join(c) for c in group.columns]
    group.reset_index().to_csv(OUT/'multiseed_summary.csv',index=False)
    pairs=[]
    for d in DATASETS:
        for b in COLORS:
            base=main[(main.dataset==d)&(main.backend==b)&(main.variant=='baseline')].set_index('seed')
            full=main[(main.dataset==d)&(main.backend==b)&(main.variant=='improved')].set_index('seed')
            delta=100*(full.loc[SEEDS,'accuracy']-base.loc[SEEDS,'accuracy'])
            pairs.append(dict(dataset=d,backend=b,mean_gain_pp=delta.mean(),sd_gain_pp=delta.std(ddof=1),
                              positive_seeds=int((delta>0).sum()),seed42_gain_pp=delta.loc[42],
                              seed7_gain_pp=delta.loc[7],seed2026_gain_pp=delta.loc[2026]))
    pd.DataFrame(pairs).to_csv(OUT/'paired_seed_gains.csv',index=False)
    ablation=[]
    for (d,a),subset in frame[frame.ablation.notna()].groupby(['dataset','ablation']):
        ab=subset.set_index('seed');full=main[(main.dataset==d)&(main.backend=='pytorch')&(main.variant=='improved')].set_index('seed')
        delta=100*(ab.loc[SEEDS,'accuracy']-full.loc[SEEDS,'accuracy'])
        ablation.append(dict(dataset=d,ablation=a,accuracy_mean=ab.accuracy.mean(),accuracy_std=ab.accuracy.std(ddof=1),
                             macro_f1_mean=ab.macro_f1.mean(),macro_f1_std=ab.macro_f1.std(ddof=1),
                             delta_vs_full_mean_pp=delta.mean(),delta_vs_full_std_pp=delta.std(ddof=1),
                             parameter_count=int(ab.parameter_count.iloc[0]),
                             seed42_delta_pp=delta.loc[42],seed7_delta_pp=delta.loc[7],seed2026_delta_pp=delta.loc[2026]))
    af=pd.DataFrame(ablation);af.to_csv(OUT/'ablation_summary.csv',index=False)
    for d in DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(10,3.35))
        for b,color in COLORS.items():
            for v,style in [('baseline','--'),('improved','-')]:
                hs=[histories[(d,b,v,s,None)] for s in SEEDS]
                for ax,metric in zip(axes,['val_loss','val_accuracy']):
                    arr=np.array([h[metric] for h in hs]);x=hs[0].epoch
                    mean=arr.mean(0);sd=arr.std(0,ddof=1)
                    ax.plot(x,mean,color=color,ls=style,label=b+' '+v)
                    ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.06)
                    ax.set(xlabel='Epoch',ylabel=metric);ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.subplots_adjust(bottom=.27,wspace=.28)
        savefig(d+'_mean_curves.png')
        data=load_data(d);names=data['class_names'];labels=data['y_test']
        bz=logits[(d,'pytorch','baseline',42,None)];fz=logits[(d,'pytorch','improved',42,None)]
        np.testing.assert_array_equal(labels,bz['labels']);np.testing.assert_array_equal(labels,fz['labels'])
        bp=bz['logits'].argmax(1);fp=fz['logits'].argmax(1)
        corrected=np.flatnonzero((bp!=labels)&(fp==labels));regressed=np.flatnonzero((bp==labels)&(fp!=labels))
        detail=dict(dataset=d,reference='PyTorch seed 42 fixed for all datasets',
                    both_correct=int(((bp==labels)&(fp==labels)).sum()),
                    corrected=len(corrected),regressed=len(regressed),both_wrong=int(((bp!=labels)&(fp!=labels)).sum()),
                    corrected_example_ids=corrected[:4].tolist(),regressed_example_ids=regressed[:4].tolist())
        (OUT/(d+'_paired_errors.json')).write_text(json.dumps(detail,indent=2),encoding='utf-8')
        fig,axes=plt.subplots(2,4,figsize=(9,4.7))
        for row,ids in enumerate([corrected[:4],regressed[:4]]):
            for col,ax in enumerate(axes[row]):
                ax.axis('off')
                if col>=len(ids):continue
                idx=ids[col];im=data['x_test'][idx]
                ax.imshow(im[0] if d=='mnist' else im.transpose(1,2,0),cmap='gray' if d=='mnist' else None)
                ax.set_title(f'#{idx} T: {names[labels[idx]]}\nB: {names[bp[idx]]} | I: {names[fp[idx]]}',fontsize=8)
        fig.subplots_adjust(hspace=.58,top=.88,bottom=.02)
        fig.suptitle('Top: corrected by improved | Bottom: regressed',fontsize=11)
        savefig(d+'_paired_errors.png')
        _,cm=classification_metrics(labels,fz['logits'],len(names))
        precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(len(names)),where=cm.sum(0)>0)
        recall=np.diag(cm)/cm.sum(1)
        pd.DataFrame(dict(label=names,support=cm.sum(1),precision=precision,recall=recall)).to_csv(OUT/(d+'_class_metrics_reference.csv'),index=False)
        if d=='cifar100':
            train_hashes={hashlib.sha256(im.tobytes()).digest() for im in data['x']}
            keep=np.array([hashlib.sha256(im.tobytes()).digest() not in train_hashes for im in data['x_test']])
            sensitivity=[]
            for b in COLORS:
                for v in ['baseline','improved']:
                    official=[];unseen=[]
                    for s in SEEDS:
                        z=logits[(d,b,v,s,None)]
                        correct=z['logits'].argmax(1)==labels
                        official.append(correct.mean());unseen.append(correct[keep].mean())
                    sensitivity.append(dict(backend=b,variant=v,official_mean=float(np.mean(official)),
                                            unseen_mean=float(np.mean(unseen)),unseen_std=float(np.std(unseen,ddof=1)),
                                            excluded=int((~keep).sum()),retained=int(keep.sum())))
            (OUT/'extended_duplicate_sensitivity.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf-8')
            raw=read_pickle(data_root()/'cifar100/train');meta=read_pickle(data_root()/'cifar100/meta')
            mapping=np.full(100,-1,dtype=int)
            for fine,coarse in zip(raw[b'fine_labels'],raw[b'coarse_labels']):
                assert mapping[fine] in [-1,coarse];mapping[fine]=coarse
            cc=np.bincount(mapping[labels]*20+mapping[fp],minlength=400).reshape(20,20)
            cn=[x.decode() for x in meta[b'coarse_label_names']]
            plt.figure(figsize=(8,6.3));plt.imshow(cc/cc.sum(1,keepdims=True),cmap='Blues',vmin=0,vmax=1)
            plt.xticks(range(20),cn,rotation=90,fontsize=12);plt.yticks(range(20),cn,fontsize=12)
            plt.xlabel('Predicted coarse group from fine prediction');plt.ylabel('True coarse group')
            plt.colorbar(fraction=.03);savefig('cifar100_coarse_confusion.png')
            np.savetxt(OUT/'cifar100_coarse_confusion.csv',cc,fmt='%d',delimiter=',')
    for d in ['cifar10','cifar100']:
        sub=af[af.dataset==d];full=main[(main.dataset==d)&(main.backend=='pytorch')&(main.variant=='improved')]
        labels=['Full','No BN','No skip','No dropout'];order=['no_bn','no_skip','no_dropout']
        means=[full.accuracy.mean()]+[float(sub[sub.ablation==a].accuracy_mean.iloc[0]) for a in order]
        stds=[full.accuracy.std(ddof=1)]+[float(sub[sub.ablation==a].accuracy_std.iloc[0]) for a in order]
        plt.figure(figsize=(8,3));plt.bar(labels,100*np.array(means),yerr=100*np.array(stds),capsize=5,color=['#206891','#bb5936','#d0a04b','#388366'])
        plt.ylabel('Test accuracy (%)');plt.ylim(0,max(means)*115);plt.grid(axis='y',alpha=.2)
        savefig(d+'_ablation.png')
    (OUT/'extended_output_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Verified 72 runs; generated seed, ablation and paired-error analyses.')

if __name__=='__main__':analyze()
