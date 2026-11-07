#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""今天誰主審？ 2026 中職逐球資料 -> 好球帶驗證 -> ML -> 歷史情蒐 -> 離線 HTML

Usage:
  python cpbl_umpire_pipeline.py --input umpire_pitch_calls.csv --out output
  python cpbl_umpire_pipeline.py --input umpire_pitch_calls.csv --out output --bootstrap 4000

Python >=3.9, pandas/numpy/scipy/scikit-learn. All derived data are local.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# Confirmed algebraic reconstruction of the THIRD-PARTY scoring dataset,
# NOT official CPBL strike-zone rule and NOT the literal baseball radius.
CORE_HALF_WIDTH_M = 0.2200
SOURCE_EXPANSION_M = 0.0369
# 2024 tested baseball diameter: 2.89 inches, not assumed to be the 2026 size.
HISTORICAL_BALL_DIAMETER_INCH = 2.89
PHYSICAL_RADIUS_M = HISTORICAL_BALL_DIAMETER_INCH * 0.0254 / 2
EDGE_THRESHOLD_CM = 10.0
CORNER_AMBIGUITY_CM = 1.0
REGIONS = ['High', 'Low', 'Left', 'Right', 'Corner']
REGION_LABEL = {'High':'上緣','Low':'下緣','Left':'座標左側','Right':'座標右側','Corner':'角落／方向模糊'}
MODELS = ['A_No_Umpire','B_With_Umpire','C_Umpire_Interaction']
C_VALUE = 1.0
SEED = 20261106
MIN_N, MIN_GAMES = 80, 10
MIN_PERIOD_N, MIN_PERIOD_GAMES = 25, 3
# Count-conditioned historical splits usually have much smaller sample sizes.
COUNT_CI_MIN_N, COUNT_CI_MIN_GAMES = 20, 4
COUNT_GRADE_MIN_N, COUNT_GRADE_MIN_GAMES = 35, 6
COUNT_PERIOD_MIN_N, COUNT_PERIOD_MIN_GAMES = 10, 2


def dump_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path,index=False,encoding='utf-8-sig')


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'找不到原始資料：{path}\n請將 umpire_pitch_calls.csv 放在同一資料夾或透過 --input 指定路徑。')
    data = pd.read_csv(path,dtype={'pitch_content':'string'},low_memory=False)
    cols=['date','game_id','umpire_name','pitch_x','pitch_z','pitch_sz_top','pitch_sz_bottom',
          'pitch_dist_cm','pitch_count_b','pitch_count_s','pitch_called','pitch_true_call']
    missing=sorted(set(cols)-set(data.columns))
    if missing:raise ValueError('原始 CSV 缺少必要欄位：'+str(missing))
    data['date']=pd.to_datetime(data['date'],errors='coerce')
    numeric=['pitch_x','pitch_z','pitch_sz_top','pitch_sz_bottom',
             'pitch_dist_cm','pitch_count_b','pitch_count_s']
    for col in numeric:data[col]=pd.to_numeric(data[col],errors='coerce')
    before=len(data)
    data=data.dropna(subset=cols)
    data=data[data['pitch_called'].isin(['BALL','STRIKE']) & data['pitch_true_call'].isin(['BALL','STRIKE'])]
    data=data[(data['pitch_sz_top']>data['pitch_sz_bottom']) & data['pitch_count_b'].between(0,3) & data['pitch_count_s'].between(0,2)]
    data=data.copy().sort_values(['date','game_id'],kind='stable').reset_index(drop=True)
    if not len(data):raise ValueError('沒有符合條件的逐球資料')
    if data.groupby('game_id')['umpire_name'].nunique().max() != 1:
        raise ValueError('同一 game_id 有多位主審，無法可靠執行比賽群集抽樣')
    data['Y']=(data['pitch_called']=='STRIKE').astype(int)
    data['z_norm']=(data['pitch_z']-data['pitch_sz_bottom'])/(data['pitch_sz_top']-data['pitch_sz_bottom'])
    data['count_state']=data['pitch_count_b'].astype(int).astype(str)+'-'+data['pitch_count_s'].astype(int).astype(str)
    print(f'[資料清理] 原始 {before:,} 筆，保留 {len(data):,} 筆；{data.game_id.nunique()} 場；{data.umpire_name.nunique()} 位主審')
    return data


def geometry_signed_distance(d: pd.DataFrame, half_width:float=CORE_HALF_WIDTH_M):
    """Signed Euclidean distance from the UNEXPANDED rounded rectangle core.
    Inside rectangle negative (distance to closest side), outside positive.
    """
    x=d['pitch_x'].to_numpy(float)
    z=d['pitch_z'].to_numpy(float)
    top=d['pitch_sz_top'].to_numpy(float)
    low=d['pitch_sz_bottom'].to_numpy(float)
    dx=np.maximum(abs(x)-half_width,0)
    dz=np.maximum.reduce([low-z,z-top,np.zeros(len(d))])
    outside=np.hypot(dx,dz)
    inside=(abs(x)<=half_width)&(z>=low)&(z<=top)
    depth=np.minimum.reduce([half_width-abs(x),z-low,top-z])
    signed=np.where(inside,-depth,outside)
    return signed,inside



def infer_geometry_from_source(d:pd.DataFrame,seed:int=SEED):
    """Infer core horizontal half-width and rounded expansion radius from
    observed pitch_dist_cm, using a randomly selected training subset of GAMES;
    assess that inferred formula on held-out games. This validates the
    previously reverse-engineered constants instead of only assuming them.
    """
    rng=np.random.default_rng(seed)
    game_ids=np.array(pd.unique(d.game_id))
    rng.shuffle(game_ids)
    ntrain=max(1,min(len(game_ids)-1,int(len(game_ids)*0.65)))
    fit_games=set(game_ids[:ntrain])
    fit=d[d.game_id.isin(fit_games)]
    val=d[~d.game_id.isin(fit_games)]
    if len(fit)>6000:fit=fit.sample(n=6000,random_state=seed)
    actual=fit.pitch_dist_cm.to_numpy(float)
    def f(theta):
        signed,_=geometry_signed_distance(fit,half_width=theta[0])
        return np.abs(signed-theta[1])*100-actual
    candidate=[]
    for init in [[.2159,PHYSICAL_RADIUS_M],[.22,.0369],[.23,.04]]:
        opt=least_squares(f,init,bounds=([.18,.02],[.26,.06]),max_nfev=200)
        candidate.append((float(np.mean(np.abs(opt.fun))),opt.x))
    best=min(candidate,key=lambda p:p[0])
    half,radius=map(float,best[1])
    signed,_=geometry_signed_distance(val,half_width=half)
    fit_distance=np.abs(signed-radius)*100
    fit_calls=np.where(signed<=radius,'STRIKE','BALL')
    result={
        'fitted_on_games':len(fit_games),
        'fitted_on_pitches':len(fit),
        'heldout_games':val.game_id.nunique(),
        'heldout_pitches':len(val),
        'inferred_core_half_width_m':half,
        'inferred_rounding_radius_m':radius,
        'heldout_distance_mae_cm':float(np.abs(fit_distance-val.pitch_dist_cm.to_numpy()).mean()),
        'heldout_call_accuracy':float((fit_calls==val.pitch_true_call.to_numpy()).mean()),
    }
    print(f'[推回幾何] 核心半寬={half:.6f} m, 圓角半徑={radius:.6f} m；'
          f'留出 {len(val):,} 顆球，理論判決吻合={result["heldout_call_accuracy"]:.6%}')
    if abs(half-CORE_HALF_WIDTH_M)>0.001 or abs(radius-SOURCE_EXPANSION_M)>0.001:
        raise ValueError('自動反推的來源幾何與已驗證設定不同，請檢查資料版本／算法。')
    return result


def add_geometry(d: pd.DataFrame):
    signed,inside_core=geometry_signed_distance(d)
    expected_distance=np.abs(signed-SOURCE_EXPANSION_M)*100
    pred_true=np.where(signed<=SOURCE_EXPANSION_M,'STRIKE','BALL')
    error=expected_distance-d['pitch_dist_cm'].to_numpy(float)
    matches=pred_true==d['pitch_true_call'].to_numpy()
    verification={
        'source_core_half_width_m':CORE_HALF_WIDTH_M,
        'source_rounded_expansion_m':SOURCE_EXPANSION_M,
        'horizontal_outer_limit_m':CORE_HALF_WIDTH_M+SOURCE_EXPANSION_M,
        'physical_ball_radius_m_2024_reference_only':PHYSICAL_RADIUS_M,
        'n_pitches':len(d),'correct_theoretical_calls':int(matches.sum()),
        'theoretical_call_agreement':float(matches.mean()),
        'distance_mae_cm':float(np.abs(error).mean()),
        'distance_max_abs_error_cm':float(np.abs(error).max()),
    }
    print(f'[好球帶驗證] 理論判決 {verification["theoretical_call_agreement"]:.6%} 一致；距離 MAE={verification["distance_mae_cm"]:.12g} cm')
    if verification['theoretical_call_agreement']<0.99999 or verification['distance_max_abs_error_cm']>1e-3:
        raise ValueError('好球帶重建沒有吻合來源。資料版本或來源演算法可能改變；已停止，避免產出錯誤情蒐卡。')
    x=d['pitch_x'].to_numpy(float)
    z=d['pitch_z'].to_numpy(float)
    top=d['pitch_sz_top'].to_numpy(float)
    bottom=d['pitch_sz_bottom'].to_numpy(float)
    bounds=np.column_stack([top-z,z-bottom,x+CORE_HALF_WIDTH_M,CORE_HALF_WIDTH_M-x])
    labels=np.array(REGIONS[:4],dtype=object)
    region=labels[np.argmin(bounds,axis=1)].astype(object)
    sorted_d=np.sort(bounds,axis=1)
    region[inside_core & ((sorted_d[:,1]-sorted_d[:,0])*100<CORNER_AMBIGUITY_CM)]='Corner'
    outside_x=abs(x)>CORE_HALF_WIDTH_M
    high=z>top;low=z<bottom;outside_z=high|low
    region[(x< -CORE_HALF_WIDTH_M)&~outside_z]='Left'
    region[(x> CORE_HALF_WIDTH_M)&~outside_z]='Right'
    region[~outside_x&high]='High'
    region[~outside_x&low]='Low'
    region[outside_x&outside_z]='Corner'
    d=d.copy()
    d['Region']=region
    d['Is_Edge']=d['pitch_dist_cm']<=EDGE_THRESHOLD_CM
    d['Theoretical_Strike']=(signed<=SOURCE_EXPANSION_M).astype(int)
    return d,verification


def one_hot():
    try:return OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    except TypeError:return OneHotEncoder(handle_unknown='ignore',sparse=True)


def make_matrices(train:pd.DataFrame,test:pd.DataFrame):
    """Three nested classifiers; all preprocessing fit ONLY on train."""
    scaler=StandardScaler()
    pt=scaler.fit_transform(train[['pitch_x','z_norm']])
    pv=scaler.transform(test[['pitch_x','z_norm']])
    poly=PolynomialFeatures(degree=2,include_bias=False)
    poly_tr=poly.fit_transform(pt);poly_te=poly.transform(pv)
    cnt=one_hot()
    ct=cnt.fit_transform(train[['count_state']]);cv=cnt.transform(test[['count_state']])
    A_tr=sparse.hstack([sparse.csr_matrix(poly_tr),ct],format='csr')
    A_te=sparse.hstack([sparse.csr_matrix(poly_te),cv],format='csr')
    ump=one_hot()
    ut=ump.fit_transform(train[['umpire_name']]);uv=ump.transform(test[['umpire_name']])
    B_tr=sparse.hstack([A_tr,ut],format='csr')
    B_te=sparse.hstack([A_te,uv],format='csr')
    def interact(U,P):
        return sparse.hstack([U.multiply(P[:,0].reshape(-1,1)),U.multiply(P[:,1].reshape(-1,1))],format='csr')
    C_tr=sparse.hstack([B_tr,interact(ut,pt)],format='csr')
    C_te=sparse.hstack([B_te,interact(uv,pv)],format='csr')
    return [(A_tr,A_te),(B_tr,B_te),(C_tr,C_te)]


def classifier():
    return LogisticRegression(C=C_VALUE,solver='lbfgs',max_iter=3000,tol=1e-6,random_state=SEED)


def pitch_loss(y,p):
    p=np.clip(np.asarray(p),1e-12,1-1e-12)
    y=np.asarray(y)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def model_comparison(d:pd.DataFrame,split:pd.Timestamp,output:Path,boot:int):
    tr=d[d['date']<split].copy().reset_index(drop=True)
    te=d[d['date']>=split].copy().reset_index(drop=True)
    if tr.empty or te.empty:raise ValueError(f'時間切分 {split.date()} 後訓練或測試集沒有資料。調整 --split。')
    if set(tr.game_id)&set(te.game_id):raise ValueError('訓練與測試集比賽重疊')
    if tr['Y'].nunique()!=2 or te['Y'].nunique()!=2:raise ValueError('訓練/測試集缺少好壞球任一類別')
    matrices=make_matrices(tr,te)
    rows=[]
    for name,(Xtr,Xte) in zip(MODELS,matrices):
        m=classifier();m.fit(Xtr,tr.Y)
        p_train=m.predict_proba(Xtr)[:,1]
        p_test=m.predict_proba(Xte)[:,1]
        te[name]=p_test
        rows.append({'Model':name,'Train_LogLoss':log_loss(tr.Y,p_train,labels=[0,1]),
                     'Test_LogLoss':log_loss(te.Y,p_test,labels=[0,1]),
                     'Test_Brier':brier_score_loss(te.Y,p_test),
                     'Test_AUC':roc_auc_score(te.Y,p_test),'Train_N':len(tr),'Test_N':len(te)})
    result=pd.DataFrame(rows)
    dump_csv(result,output/'model_ABC_time_holdout.csv')
    dump_csv(te[['date','game_id','umpire_name','Region','Is_Edge','pitch_dist_cm','Y']+MODELS],output/'model_ABC_test_predictions.csv')
    # Fair aggregate comparison: games are the resampling units.
    rng=np.random.default_rng(SEED+1)
    games=np.asarray(pd.unique(te.game_id))
    sample=rng.integers(0,len(games),size=(boot,len(games)))
    test_rows=[]
    for scope,part in [('All_Pitches',te),('Edge_10cm',te[te.Is_Edge])]:
        for base,candidate in [('A_No_Umpire','B_With_Umpire'),('A_No_Umpire','C_Umpire_Interaction'),('B_With_Umpire','C_Umpire_Interaction')]:
            gain=pitch_loss(part.Y,part[base])-pitch_loss(part.Y,part[candidate])
            gs=pd.DataFrame({'game_id':part.game_id.to_numpy(),'gain':gain}).groupby('game_id').gain.agg(['sum','count']).reindex(games,fill_value=0)
            counts=gs['count'].to_numpy(float)[sample].sum(axis=1)
            sums=gs['sum'].to_numpy(float)[sample].sum(axis=1)
            bs=np.divide(sums,counts,out=np.full(len(counts),np.nan),where=counts>0)
            lo,hi=np.nanpercentile(bs,[2.5,97.5])
            test_rows.append({'Scope':scope,'Baseline':base,'Candidate':candidate,'Pitches':len(part),'Games':part.game_id.nunique(),
                              'LogLoss_Gain':gain.mean(),'CI_Low':lo,'CI_High':hi,'CI_Excludes_Zero':bool(lo>0 or hi<0)})
    test_boot=pd.DataFrame(test_rows)
    dump_csv(test_boot,output/'model_ABC_game_bootstrap.csv')
    print('[樣本外模型] '+', '.join(f'{r.Model}: {r.Test_LogLoss:.5f}' for r in result.itertuples()))
    print(f'[時間切分] {split.date()}；train={len(tr)}, test={len(te)}，test 場次={te.game_id.nunique()}')
    return result,test_boot


def oof_expected_rate(d:pd.DataFrame,n_folds:int=5):
    """For HISTORICAL descriptions only: game-grouped cross-fitting of Model A.
    Not a time-forward prediction and not umpire-specific.
    """
    unique_games=d.game_id.nunique()
    n_folds=min(n_folds,unique_games)
    if n_folds<2:raise ValueError('至少需要兩場比賽才能交叉驗證')
    result=np.full(len(d),np.nan)
    splitter=GroupKFold(n_splits=n_folds)
    for fold,(it,iv) in enumerate(splitter.split(d,d.Y,groups=d.game_id),start=1):
        tr=d.iloc[it];te=d.iloc[iv]
        (Xt,Xv),_,_=make_matrices(tr,te)
        clf=classifier();clf.fit(Xt,tr.Y)
        result[iv]=clf.predict_proba(Xv)[:,1]
        print(f'[歷史基準] GroupKFold {fold}/{n_folds}: {len(iv)} 球')
    assert np.isfinite(result).all()
    return result


def bhs(p):
    """BH-adjusted p values; handle missing p with NaN."""
    p=np.asarray(p,float)
    out=np.full(len(p),np.nan)
    ix=np.flatnonzero(np.isfinite(p))
    if not len(ix):return out
    ranks=ix[np.argsort(p[ix])]
    m=len(ix)
    qq=np.minimum.accumulate((p[ranks]*m/np.arange(1,m+1))[::-1])[::-1]
    out[ranks]=np.clip(qq,0,1)
    return out


def bootstrap_game_mean(part:pd.DataFrame,n_boot:int,rng):
    if len(part)==0:return np.full(n_boot,np.nan)
    groups=part.groupby('game_id')['Residual'].agg(['sum','count']).to_numpy(float)
    k=len(groups)
    inds=rng.integers(0,k,size=(n_boot,k))
    s=groups[inds,0].sum(axis=1)
    n=groups[inds,1].sum(axis=1)
    return np.divide(s,n,out=np.full(n_boot,np.nan),where=n>0)


def relative_historical(d:pd.DataFrame,output:Path,n_boot:int):
    d=d.copy()
    d['Expected_Strike']=oof_expected_rate(d)
    d['Residual']=d.Y-d.Expected_Strike
    edge=d[d.Is_Edge].copy()
    # Split by game date, keeping all games on the same date together.
    dates=edge[['game_id','date']].drop_duplicates('game_id')['date'].sort_values()
    mid_date=dates.iloc[len(dates)//2]
    edge['Period']=np.where(edge.date<mid_date,'First_Half','Second_Half')
    print(f'[歷史資料] edge={len(edge)}；半季切點={mid_date.date()}')
    dump_csv(edge[['date','game_id','umpire_name','Region','pitch_count_b','pitch_count_s','count_state','pitch_dist_cm','Y','Expected_Strike','Residual','Period']],output/'historical_edge_pitches_oof.csv')
    names=sorted(edge.umpire_name.unique())
    # Relative difference: each umpire against all the OTHER umpires in the same region.
    def compare(df,umpire,region):
        a=df[(df.umpire_name==umpire)&(df.Region==region)]
        b=df[(df.umpire_name!=umpire)&(df.Region==region)]
        if a.empty or b.empty:return None
        return {'N':len(a),'Games':a.game_id.nunique(),'Other_N':len(b),'Other_Games':b.game_id.nunique(),
                'Actual_Strike_pct':100*a.Y.mean(),'Expected_Strike_pct':100*a.Expected_Strike.mean(),
                'Own_Residual_pp':100*a.Residual.mean(),'Others_Residual_pp':100*b.Residual.mean(),
                'Difference_pp':100*(a.Residual.mean()-b.Residual.mean())}
    rows=[]
    rng=np.random.default_rng(SEED+5)
    for i,umpire in enumerate(names):
        for reg in REGIONS:
            x=compare(edge,umpire,reg)
            if x is None:continue
            own=edge[(edge.umpire_name==umpire)&(edge.Region==reg)]
            other=edge[(edge.umpire_name!=umpire)&(edge.Region==reg)]
            own_bs=bootstrap_game_mean(own,n_boot,rng)
            other_bs=bootstrap_game_mean(other,n_boot,rng)
            z=(own_bs-other_bs)*100
            lo,hi=np.nanpercentile(z,[2.5,97.5])
            obs=x['Difference_pp']
            # Centered-bootstrap approximate two-sided p; OOF probabilities are treated as fixed.
            null=z-obs
            valid=null[np.isfinite(null)]
            p=(1+np.count_nonzero(abs(valid)>=abs(obs)))/(1+len(valid))
            periods={}
            for period in ['First_Half','Second_Half']:
                sub=edge[edge.Period==period]
                v=compare(sub,umpire,reg)
                for key in ['N','Games','Difference_pp']:
                    periods[f'{key}_{period}']=v[key] if v else np.nan
            rows.append({'Umpire':umpire,'Region':reg,**x,
                         'CI_Low_pp':lo,'CI_High_pp':hi,'P_CenteredBootstrap':p,**periods})
    table=pd.DataFrame(rows)
    table['Q_BH']=bhs(table.P_CenteredBootstrap.to_numpy())
    table['Sample_Adequate']=(table.N>=MIN_N)&(table.Games>=MIN_GAMES)
    table['Period_Adequate']=(table.N_First_Half>=MIN_PERIOD_N)&(table.N_Second_Half>=MIN_PERIOD_N)&(table.Games_First_Half>=MIN_PERIOD_GAMES)&(table.Games_Second_Half>=MIN_PERIOD_GAMES)
    table['Same_Sign_Relative']=(table.Difference_pp_First_Half*table.Difference_pp_Second_Half>0)
    table['CrossPeriod_Stable']=table.Period_Adequate&table.Same_Sign_Relative
    # No A-D grades. A misleading apparent "D" previously arose even for
    # tiny sign changes; E is ONLY an explicit sample coverage warning.
    table['Sample_Warning_E'] = (~table.Sample_Adequate | ~table.Period_Adequate)
    table['Sample_Notice'] = np.where(table.Sample_Warning_E, 'E｜歷史樣本不足', '樣本符合門檻（不代表偏好已獲證實）')
    order={k:i for i,k in enumerate(REGIONS)}
    table['region_order']=table.Region.map(order)
    table=table.sort_values(['Umpire','region_order']).drop(columns=['region_order']).reset_index(drop=True)
    dump_csv(table,output/'umpire_region_relative_with_sample_warning.csv')
    overall=edge.groupby('umpire_name').agg(N=('Y','size'),Games=('game_id','nunique'),Actual=('Y','mean'),Expected=('Expected_Strike','mean'),Residual=('Residual','mean')).reset_index().rename(columns={'umpire_name':'Umpire'})
    overall[['Actual','Expected','Residual']]*=100
    overall_warning=table.groupby('Umpire')['Sample_Warning_E'].sum().rename('E_Warnings').reset_index()
    overall=overall.merge(overall_warning,on='Umpire',how='left')
    dump_csv(overall,output/'umpire_historical_summary.csv')
    print(f'[歷史樣本提醒] {int(table.Sample_Warning_E.sum())}/{len(table)} 個主審×區域組合顯示 E｜樣本不足；其餘不分級。')
    return table,overall,len(edge),mid_date,edge




def count_conditioned_historical(edge:pd.DataFrame,output:Path,n_boot:int):
    """EXPLORATORY historical splits for each recorded ball/strike count.

    A selected count is always based on actual historical pitches at that count,
    never a counterfactual prediction. OOF Model A (including count one-hot)
    supplies the expected strike probability. The comparison pool is other
    umpires at the *same count AND same region*. Small cells are disclosed.

    95% CI is game-cluster bootstrap, only for cells passing a minimal sampling
    check; no q values are invented for selected-count results, and historical
    no A-D grades are assigned, for any count or full-season subset.
    """
    states=[f'{b}-{st}' for b in range(4) for st in range(3)]
    names=sorted(edge.umpire_name.unique())
    count_overall=(edge.groupby(['umpire_name','count_state'])
       .agg(N=('Y','size'),Games=('game_id','nunique'),Actual=('Y','mean'),
            Expected=('Expected_Strike','mean'),Residual=('Residual','mean'))
       .reset_index().rename(columns={'umpire_name':'Umpire','count_state':'Count'}))
    count_overall[['Actual','Expected','Residual']]*=100
    dump_csv(count_overall,output/'umpire_count_overall.csv')
    rows=[]
    rng=np.random.default_rng(SEED+71)
    n_boot=min(n_boot,1500)
    for state in states:
        subset=edge[edge.count_state==state]
        for region in REGIONS:
            sub=subset[subset.Region==region]
            if sub.empty:continue
            # Compute same-count, same-region comparison against everyone else.
            n_total=len(sub); sum_total=sub.Residual.sum()
            for umpire in names:
                own=sub[sub.umpire_name==umpire]
                n=len(own)
                others_n=n_total-n
                if n==0 or others_n==0:
                    continue
                other_sum=sum_total-own.Residual.sum()
                own_mean=own.Residual.mean()
                others_mean=other_sum/others_n
                diff=(own_mean-others_mean)*100
                n_games=own.game_id.nunique()
                other=sub[sub.umpire_name!=umpire]
                per=[]
                for phase in ['First_Half','Second_Half']:
                    phase_sub=sub[sub.Period==phase]
                    a=phase_sub[phase_sub.umpire_name==umpire]
                    b=phase_sub[phase_sub.umpire_name!=umpire]
                    per.append({'n':len(a),'games':a.game_id.nunique(),
                        'diff':(a.Residual.mean()-b.Residual.mean())*100
                        if len(a)>0 and len(b)>0 else np.nan})
                period_adequate=all(x['n']>=COUNT_PERIOD_MIN_N and x['games']>=COUNT_PERIOD_MIN_GAMES for x in per)
                same_sign=bool(period_adequate and per[0]['diff']*per[1]['diff']>0)
                sample_adequate=n>=COUNT_GRADE_MIN_N and n_games>=COUNT_GRADE_MIN_GAMES
                # Count-specific data are exploratory, never assigned A-D.
                sample_status=('較充足' if sample_adequate and period_adequate
                              else '有限' if n>=COUNT_CI_MIN_N and n_games>=COUNT_CI_MIN_GAMES
                              else '不足')
                low=high=np.nan
                if n>=COUNT_CI_MIN_N and n_games>=COUNT_CI_MIN_GAMES and other.game_id.nunique()>=5:
                    a_bs=bootstrap_game_mean(own,n_boot,rng)
                    b_bs=bootstrap_game_mean(other,n_boot,rng)
                    bs=(a_bs-b_bs)*100
                    if np.isfinite(bs).sum()>=int(n_boot*.9):
                        low,high=np.nanpercentile(bs,[2.5,97.5])
                rows.append({
                    'Umpire':umpire,'Count':state,'Region':region,
                    'N':n,'Games':n_games,'Other_N':others_n,
                    'Actual_Strike_pct':own.Y.mean()*100,
                    'Expected_Strike_pct':own.Expected_Strike.mean()*100,
                    'Difference_pp':diff,'CI_Low_pp':low,'CI_High_pp':high,
                    'Difference_pp_First_Half':per[0]['diff'],
                    'Difference_pp_Second_Half':per[1]['diff'],
                    'N_First_Half':per[0]['n'],'N_Second_Half':per[1]['n'],
                    'Games_First_Half':per[0]['games'],'Games_Second_Half':per[1]['games'],
                    'Sample_Adequate':bool(sample_adequate),
                    'Period_Adequate':bool(period_adequate),
                    'Same_Sign_Relative':same_sign,
                    'Count_Sample_Status':sample_status,
                    'Sample_Warning_E':bool(sample_status=='不足'),
                    'Sample_Notice': 'E｜指定球數樣本不足' if sample_status=='不足' else '指定球數探索性描述',
                    'Count_Inference_Only':True,
                })
    conditioned=pd.DataFrame(rows)
    dump_csv(conditioned,output/'umpire_region_by_count_exploratory.csv')
    count_sizes=edge.groupby('count_state').agg(Pitches=('Y','size'),Games=('game_id','nunique')).reset_index().rename(columns={'count_state':'Count'})
    dump_csv(count_sizes,output/'count_state_sample_sizes.csv')
    print('[不同球數] '+', '.join(f'{r.Count}:{r.Pitches}' for r in count_sizes.itertuples()))
    print('[不同球數] 僅呈現描述性樣本狀態，樣本不足提示 E，不進行 A-D 等級評分。')
    return conditioned,count_overall,count_sizes


RANK_MIN_RAW_N, RANK_MIN_RAW_GAMES = 30, 5
RANK_MIN_THEORY_N, RANK_MIN_THEORY_GAMES = 20, 4

def league_rankings(d:pd.DataFrame,output:Path)->pd.DataFrame:
    """Actual, observed called-strike rankings, not inferred umpire 'intent'.

    1) Actual strike rate: STRIKE / all edge-zone pitches.
    2) False-strike rate: STRIKE / theoretical BALL edge pitches.
    3) Missed-strike rate: BALL / theoretical STRIKE edge pitches.

    A theoretical BALL is defined ONLY by the already VERIFIED third-party
    geometric `pitch_true_call`, NOT official league accuracy.
    """
    edge=d[d.Is_Edge].copy()
    edge['Is_FalseStrike']=((edge.Theoretical_Strike==0)&(edge.Y==1)).astype(int)
    edge['Is_TrueBall']=(edge.Theoretical_Strike==0).astype(int)
    edge['Is_MissedStrike']=((edge.Theoretical_Strike==1)&(edge.Y==0)).astype(int)
    edge['Is_TrueStrike']=(edge.Theoretical_Strike==1).astype(int)
    out=[]
    states=['ALL']+[f'{b}-{st}' for b in range(4) for st in range(3)]
    for state in states:
        sub=edge if state=='ALL' else edge[edge.count_state==state]
        for (u,r), part in sub.groupby(['umpire_name','Region'],sort=True):
            n=len(part);games=part.game_id.nunique()
            true_ball=part[part.Is_TrueBall==1]
            true_strike=part[part.Is_TrueStrike==1]
            nball=len(true_ball); nstrike=len(true_strike)
            gb=true_ball.game_id.nunique(); gs=true_strike.game_id.nunique()
            fs=int(part.Is_FalseStrike.sum()); ms=int(part.Is_MissedStrike.sum())
            sr=int(part.Y.sum())
            ready_raw=(n>=RANK_MIN_RAW_N and games>=RANK_MIN_RAW_GAMES)
            ready_ball=(nball>=RANK_MIN_THEORY_N and gb>=RANK_MIN_THEORY_GAMES)
            ready_strike=(nstrike>=RANK_MIN_THEORY_N and gs>=RANK_MIN_THEORY_GAMES)
            out.append({
                'Count':state, 'Umpire':u, 'Region':r,
                'N_Edge':n, 'Games':games,
                'Called_Strike_N':sr,
                'Called_Strike_Rate_pct':100*sr/n,
                'Called_Strike_Eligible':bool(ready_raw),
                'Theoretical_Ball_N':nball, 'Theoretical_Ball_Games':gb,
                'False_Strike_N':fs,
                'False_Strike_Rate_pct':100*fs/nball if nball else np.nan,
                'False_Strike_Eligible':bool(ready_ball),
                'Theoretical_Strike_N':nstrike, 'Theoretical_Strike_Games':gs,
                'Missed_Strike_N':ms,
                'Missed_Strike_Rate_pct':100*ms/nstrike if nstrike else np.nan,
                'Missed_Strike_Eligible':bool(ready_strike),
            })
    table=pd.DataFrame(out)
    if table.empty:raise ValueError('沒有足夠邊緣球可建立排行榜')
    dump_csv(table,output/'umpire_league_rankings.csv')
    print(f'[聯盟排行榜] {len(table)} 組球數×主審×區域，理論壞球誤判好球與實際好球率均可排序。')
    return table

def json_safe(x):
    """JSON safely embedded into a standalone <script> tag."""
    return json.dumps(x,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')


def as_records(df:pd.DataFrame):
    # Cast missing to null for JSON (never NaN or Inf).
    return json.loads(df.to_json(orient='records',force_ascii=False,date_format='iso'))




def static_mobile_fallback(payload: dict) -> str:
    '''Pre-render a useful HTML snapshot for iOS Quick Look / JS-blocked viewers.

    The interactive renderer replaces this section only AFTER it successfully renders.
    Snapshot remains visible if scripts are disabled or encounter a runtime error.
    '''
    from html import escape as esc

    regs = ['High','Low','Left','Right','Corner']
    labels = {'High':'上緣','Low':'下緣','Left':'座標左側','Right':'座標右側','Corner':'角落'}
    season = {(r['Umpire'],r['Region']):r for r in payload.get('regions',[])}
    count = {(r['Umpire'],r['Region']):r for r in payload.get('count_regions',[])
             if r.get('Count')=='0-0'}
    all_names=sorted({r['Umpire'] for r in payload.get('regions',[])})
    names=[n for n in ['陳均瑋','吳家維'] if n in all_names]
    names+= [n for n in all_names if n not in names]
    names=names[:2]

    def fmt(x, unit=''):
        if x is None: return '—'
        try:
            v=float(x)
            if not math.isfinite(v):return '—'
            return f'{v:+.1f}{unit}'
        except (ValueError,TypeError):
            return '—'

    panels=[]
    for name in names:
        lines=[]
        for reg in regs:
            r=count.get((name,reg),{})
            season_r=season.get((name,reg),{})
            n=r.get('N',0) or 0
            games=r.get('Games',0) or 0
            value=fmt(r.get('Difference_pp'),' pp') if n>=10 else '—（少於10球）'
            quality=('⚠ E｜全季樣本不足' if season_r.get('Sample_Warning_E') else '全季樣本符合門檻')
            lines.append(f'<tr><td>{labels[reg]}</td><td>{value}</td><td>{n}球／{games}場</td><td>{quality}</td></tr>')
        panels.append(f'<div class="introitem"><h3>{esc(name)}｜0壞0好</h3>'
                      '<p>與同區域、同球數其他主審的歷史差異（百分點），已扣除模型預期。</p>'
                      '<div class="tablewrap"><table class="leaderboard"><thead><tr>'
                      '<th>區域</th><th>差異</th><th>樣本</th><th>全季提醒</th>'
                      '</tr></thead><tbody>'+''.join(lines)+'</tbody></table></div></div>')

    ranking = [r for r in payload.get('rankings',[])
               if r.get('Count')=='ALL' and r.get('Region')=='Left']
    ranking.sort(key=lambda r:(not bool(r.get('False_Strike_Eligible')),
                               -(float(r.get('False_Strike_Rate_pct') or 0))))
    rank_lines=[]
    rank_num=0
    for r in ranking:
        eligible=bool(r.get('False_Strike_Eligible'))
        if eligible:rank_num+=1
        val=r.get('False_Strike_Rate_pct')
        rate=(f'{float(val):.1f}%' if val is not None else '—')
        rank_lines.append('<tr>'
           f'<td>{rank_num if eligible else "—"}</td>'
           f'<td>{esc(str(r.get("Umpire","")))}</td>'
           f'<td>{rate}</td>'
           f'<td>{r.get("False_Strike_N",0) or 0}／{r.get("Theoretical_Ball_N",0) or 0}</td>'
           f'<td>{r.get("Theoretical_Ball_Games",0) or 0}</td>'
           f'<td>{"達門檻" if eligible else "⚠ E｜樣本不足"}</td>'
           '</tr>')

    is_synthetic='合成' in str(payload.get('preview_label',''))
    disclaimer=('⚠ 這是合成測試數據，姓名僅用於介面展示，非真實中職判決。' if is_synthetic
                else '本頁使用原始 CSV 計算的歷史觀察資料。')
    return f'''<section id="staticMobileFallback" class="panel" style="border:2px solid #6ba4a1;background:#183148">
      <h2>📱 手機預覽模式｜已載入的靜態資料</h2>
      <p><strong>這份檔案的資料已包含在 HTML 裡。</strong> 如果下方互動圖表空白，代表目前的檔案預覽器未執行 JavaScript。</p>
      <p class="small">請把 HTML 放上支援靜態網站的服務，再用 Safari 開啟公開或私人網址，才能切換主審／球數與排行榜。此處先提供預設畫面的實際嵌入資料。</p>
      <p style="color:#ffcc9d;font-weight:700">{esc(disclaimer)}</p>
      <h3>預設主審情蒐卡｜0壞0好</h3>
      <div class="cols3" style="grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))">{''.join(panels)}</div>
      <h3 style="margin-top:18px">聯盟排行榜｜全季所有球數 · 座標左側 · 理論壞球被判好球率</h3>
      <p class="small">分母是該區域的理論壞球；E 只表示樣本不足，未列入正式排名。</p>
      <div class="tablewrap"><table class="leaderboard"><thead><tr><th>排名</th><th>主審</th><th>比例</th><th>分子／分母</th><th>場次</th><th>樣本</th></tr></thead>
      <tbody>{''.join(rank_lines)}</tbody></table></div>
      </section>'''


def standalone_mobile_html(payload: dict) -> str:
    '''Script-free mobile reading edition generated from the same analysis data.'''
    from html import escape as esc
    regions=['High','Low','Left','Right','Corner']
    labels={'High':'上緣 High','Low':'下緣 Low','Left':'座標左側 Left','Right':'座標右側 Right','Corner':'角落 Corner'}
    names=sorted({r['Umpire'] for r in payload.get('regions',[])})
    full={(r['Umpire'],r['Region']):r for r in payload.get('regions',[])}
    counts={(r['Umpire'],r['Region']):r for r in payload.get('count_regions',[])
            if r.get('Count')=='0-0'}
    rankings=payload.get('rankings',[])
    is_synth='合成' in str(payload.get('preview_label',''))
    disclaimer=('⚠ 此頁為合成測試數據，真實主審姓名只是選單示範，比例與排名並非中職實況。' if is_synth
                else '分析依第三方來源的好球帶理論公式，並非官方判決準確率或主審能力評分。')
    def f(x,dec=1,sign=False):
        try:
            v=float(x)
            if not math.isfinite(v):return '—'
            return (f'{v:+.{dec}f}' if sign else f'{v:.{dec}f}')
        except (ValueError,TypeError):return '—'
    def rate_ranking(region):
        table=[r for r in rankings if r.get('Count')=='ALL' and r.get('Region')==region]
        table.sort(key=lambda r:(not bool(r.get('False_Strike_Eligible')),
                                 -float(r.get('False_Strike_Rate_pct') or 0)))
        ix=0;trs=[]
        for r in table:
            ok=bool(r.get('False_Strike_Eligible'))
            if ok:ix+=1
            name=str(r['Umpire'])
            trs.append('<tr>'+''.join([
                f'<td>{ix if ok else "—"}</td>',
                f'<td><a href="#ump-{esc(name,quote=True)}">{esc(name)}</a></td>',
                f'<td><strong>{f(r.get("False_Strike_Rate_pct"))}%</strong></td>',
                f'<td>{r.get("False_Strike_N",0) or 0}／{r.get("Theoretical_Ball_N",0) or 0}</td>',
                f'<td>{r.get("Theoretical_Ball_Games",0) or 0}</td>',
                f'<td>{"樣本達門檻" if ok else "⚠ E｜樣本不足"}</td>'
            ])+'</tr>')
        return f'<section id="region-{region}"><h2>{labels[region]}｜理論壞球被判好球率排行</h2><p>全季球數彙總，分母僅含第三方理論壞球。E 不給正式名次。</p><div class="scroll"><table><thead><tr><th>排名</th><th>主審</th><th>比例</th><th>分子／分母</th><th>場次</th><th>警示</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div></section>'
    def umpire_card(name):
        lines=[]
        for region in regions:
            r=counts.get((name,region),{})
            seasonal=full.get((name,region),{})
            n=r.get('N',0) or 0
            num=f(r.get('Difference_pp'),sign=True)+' pp' if n>=10 else '—（少於10球）'
            e='⚠ E｜全季樣本不足' if seasonal.get('Sample_Warning_E') else '全季樣本達門檻'
            lines.append('<tr>'+''.join([f'<td>{labels[region]}</td>',f'<td>{num}</td>',f'<td>{n}球／{r.get("Games",0) or 0}場</td>',f'<td>{e}</td>'])+'</tr>')
        return f'<section id="ump-{esc(name,quote=True)}"><h2>主審：{esc(name)}｜0壞0好</h2><p>歷史相對偏好（pp）：與相同球數、同區域其他主審的模型殘差比較。</p><div class="scroll"><table><thead><tr><th>區域</th><th>相對差異</th><th>球／場</th><th>全季樣本</th></tr></thead><tbody>{"".join(lines)}</tbody></table></div></section>'
    links=' '.join(f'<a class="nav" href="#region-{r}">{labels[r]}</a>' for r in regions)
    umps=' '.join(f'<a class="nav" href="#ump-{esc(n,quote=True)}">{esc(n)}</a>' for n in names)
    markup=f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>今天誰主審？｜手機純靜態版</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#101c30;color:#edf5fd;font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang TC",sans-serif}}main{{max-width:980px;margin:auto;padding:20px 13px 60px}}h1{{font-size:30px}}h2{{font-size:20px;margin:0 0 8px}}section{{background:#1b2d46;border:1px solid #44607f;border-radius:14px;margin:17px 0;padding:15px}}p{{color:#c6d4e5}}.warn{{border-left:5px solid #e4ad72;background:#3c3132;padding:12px;border-radius:8px}}.nav{{display:inline-block;padding:7px 10px;border-radius:8px;background:#29445f;margin:4px;color:#e6f5ff;text-decoration:none}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;min-width:550px}}th,td{{text-align:left;border-bottom:1px solid #39506a;padding:9px 6px}}th{{color:#a8d2df;white-space:nowrap}}td a{{color:#a8d2ee}}strong{{color:#fff}}small{{color:#b3c5d5}}
</style></head><body><main><h1>今天誰主審？</h1><p>📱 手機純靜態閱讀版｜不需要 JavaScript，可在手機檔案預覽器閱讀。注意：本版本不支援即時切換球數與排序；完整互動功能請開啟網頁版。</p><div class="warn">{esc(disclaimer)}</div><section><h2>聯盟排行榜：選區域</h2>{links}<h2 style="margin-top:16px">主審 0 壞 0 好歷史卡</h2>{umps}</section>{''.join(rate_ranking(r) for r in regions)}{''.join(umpire_card(n) for n in names)}<small>統計定義：10公分以內邊緣球；Left/Right 是資料座標，不能直接視作打者內外角；「撿理論壞球」不等於官方認定誤判；E 只表示樣本不足。</small></main></body></html>'''
    return markup

def write_html(geometry:dict,model:pd.DataFrame,test_boot:pd.DataFrame,regions:pd.DataFrame,overall:pd.DataFrame,
               pitches:int,games:int,umcount:int,edge_count:int,out:Path,preview_label:str='',
               count_regions=None,count_overall=None,count_sizes=None,rankings=None):
    body={'geometry':geometry,'models':as_records(model),'test_boot':as_records(test_boot),
          'regions':as_records(regions),'overall':as_records(overall),
          'count_regions':as_records(count_regions) if count_regions is not None else [],
          'count_overall':as_records(count_overall) if count_overall is not None else [],
          'count_sizes':as_records(count_sizes) if count_sizes is not None else [],
          'rankings':as_records(rankings) if rankings is not None else [],
          'counts':{'pitches':pitches,'games':games,'umpires':umcount,'edge':edge_count},
          'preview_label':preview_label}
    payload=json_safe(body)
    static_fallback=static_mobile_fallback(body)
    template=r'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>今天誰主審？｜中職主審歷史情蒐</title>
<style>
:root{--bg:#0c1423;--panel:#17243a;--panel2:#1e3048;--line:#34475e;--ink:#f0f6fd;--muted:#b4c4d5;--pos:#f1a080;--neg:#82baf1;--accent:#79c5b6}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% -10%,#28405e,transparent 42%),var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,'PingFang TC','Microsoft JhengHei',sans-serif}
main{max-width:1250px;margin:auto;padding:34px 22px 70px}h1{margin:2px 0;font-size:36px}h2{font-size:21px;margin:0 0 10px}h3{font-size:16px;margin:0 0 8px}p{margin:5px 0 10px}.muted,.small{color:var(--muted)}.small{font-size:12px}.kicker{font-size:12px;font-weight:800;letter-spacing:.2em;color:#88cbdd}
.top{display:flex;justify-content:space-between;align-items:start;gap:20px;flex-wrap:wrap;margin-bottom:22px}.chip{border:1px solid var(--line);padding:7px 12px;border-radius:99px;font-size:12px;color:var(--muted)}
.panel,.ump-card,.notice{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:15px 0}.notice{background:#24374d;border-left:4px solid var(--accent);font-size:14px}.notice strong{color:#e9fff8}#previewNote{display:block;margin-top:8px;color:#ffbd80;font-weight:850;font-size:15px}
.cols3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.introitem{border:1px solid var(--line);border-radius:12px;padding:14px;background:#1b2b41}.introitem strong{display:block;margin-bottom:6px}.introitem p{font-size:13px;color:var(--muted)}
.metricrow{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:14px}.metric{padding:11px 10px;background:var(--panel2);border-radius:10px}.metric b{display:block;font-size:22px;line-height:1.4}.metric span{font-size:11px;color:var(--muted)}
.countbox{border:1px solid #407e7c;border-radius:14px;background:#142f38;padding:15px;margin:16px 0}.countbox .controls{margin:9px 0}.countnote{padding:9px 12px;border:1px solid #3b6765;border-radius:10px;color:#e5f2ee;background:#193a41}.countnote.warning{border-color:#b98959;background:#3e3230;color:#ffe4c7}.disclaimer{font-size:12px;color:var(--muted)}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:15px 0}.controls label{display:block;color:var(--muted);font-size:12px;font-weight:700;margin-bottom:6px}select,button{font:inherit;border-radius:10px;padding:9px 12px;border:1px solid #58728f;background:#203149;color:var(--ink)}button{cursor:pointer}button:hover,select:hover{border-color:var(--accent)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.ump-card{margin:0}.name{font-size:25px;font-weight:800}.cardhead{display:flex;align-items:start;justify-content:space-between;gap:10px}.summaryline{font-size:13px;color:var(--muted)}
.zonegrid{display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:124px 124px 124px;gap:8px;margin:16px auto;max-width:455px}.zone{border-radius:12px;border:1px solid #6b8299;padding:10px;display:flex;flex-direction:column;gap:2px;cursor:pointer;background:#344759;text-align:center;justify-content:center}.zone:hover,.zone.active{outline:2px solid #f0dfbb}.zone.positive{background:#563b40;border-color:#a56e6c}.zone.negative{background:#254360;border-color:#668fbe}.zone.high{grid-area:1/2}.zone.low{grid-area:3/2}.zone.left{grid-area:2/1}.zone.right{grid-area:2/3}.zone.corner{grid-area:2/2;background:#304255}.zone .zn{font-size:12px;color:#e4edf6}.zone .val{font-size:23px;font-weight:800}.zone .n{font-size:10px;color:#dce6ef}.zone .evidence{font-size:10px;color:#fff0cc;line-height:1.25}.zone .sample{font-size:10px;color:#b4ede4;line-height:1.2}.evidencepair{display:flex;flex-direction:column;align-items:flex-end;gap:6px;min-width:166px}.evidencepair .pill{white-space:normal;text-align:right}.pill.countpill{background:#18423f;color:#c9ede6}.dualhint{background:#1b3245;border:1px solid #466b7a;border-radius:12px;padding:12px;margin:10px 0;font-size:13px}.dualhint b{color:#e6f2c1}
.cardtable{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 5px;border-bottom:1px solid var(--line);text-align:left}th{color:#c6d5e6}td.num{text-align:right;white-space:nowrap}.tag{display:inline-block;border:1px solid #859eb3;padding:2px 5px;border-radius:6px;white-space:normal;font-size:11px}.tag.A{border-color:#83bbaa;color:#a5e1d1}.tag.B{border-color:#b7ba8c;color:#e9e6a5}.tag.D{border-color:#d2a173;color:#eac09a}.tag.E{border-color:#838f9b;color:#cad5dc}
footer{font-size:12px;color:var(--muted);margin-top:20px}.splitline{height:1px;background:var(--line);margin:15px 0}.comparison{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-top:16px}.chart{width:100%;min-height:240px}.explain{background:#142239;border:1px solid var(--line);border-radius:14px;padding:14px;margin-top:12px}details summary{cursor:pointer;font-weight:700;padding:4px}.glossary{display:grid;grid-template-columns:160px 1fr;gap:8px 12px;font-size:13px;margin-top:12px}.glossary dt{font-weight:800}.glossary dd{margin:0;color:var(--muted)}
.tablewrap{overflow:auto}table.model{border-collapse:collapse;width:100%;font-size:13px}table.model th,table.model td{padding:9px;border-bottom:1px solid var(--line)}.pill{display:inline-block;background:#25465b;color:#b3e1dc;padding:3px 9px;border-radius:100px;font-size:11px}.subhead{color:#dce7f5;font-weight:700;margin-top:12px}svg text{font-family:inherit}
@media(max-width:940px){.cards,.cols3{grid-template-columns:1fr}.metricrow{grid-template-columns:repeat(2,1fr)}main{padding:22px 12px 42px}h1{font-size:29px}.ump-card{padding:14px}.glossary{grid-template-columns:1fr}}
@media print{body{background:white;color:#111}.panel,.ump-card,.comparison{background:white;color:#111;border-color:#aaa}.controls button{display:none}.cards{grid-template-columns:1fr 1fr}.muted,.small{color:#444}}

.ranking{background:linear-gradient(140deg,#183048,#13243b);border:1px solid #407084;border-radius:18px;padding:22px;margin:18px 0}.rankhead{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}.rankinfo{border:1px solid #45637a;background:#1c3349;padding:11px;border-radius:10px;font-size:13px;margin-bottom:14px}.leaderboard{border-collapse:collapse;width:100%;min-width:650px}.leaderboard th,.leaderboard td{border-bottom:1px solid #35516b;padding:11px 9px;text-align:left}.leaderboard tbody tr:hover{background:#28445c}.leaderboard tr.warnrow{color:#b7c3cf}.leaderboard tr.warnrow td{opacity:.9}.leaderboard td.value{font-weight:800;font-variant-numeric:tabular-nums}.leaderboard td.value.strong{color:#efc3a9}.leaderboard button{padding:5px 9px;font-size:12px}.rankwarn{font-size:11px;color:#ffe0b8;border:1px solid #8c7765;border-radius:7px;padding:3px 6px;white-space:nowrap}
</style></head><body><main>
<div class="top"><div><div class="kicker">CPBL / HISTORICAL UMPIRE SCOUTING</div><h1>今天誰主審？</h1><p class="muted">中職主審歷史判決情蒐平台 · 2026 第三方逐球資料</p></div><span class="chip" id="sampleChip"></span></div>
<div class="notice"><strong>本平台只展示歷史判決，不對主審能力評分。</strong><div class="small">只有資料涵蓋未達門檻時顯示「⚠ E｜樣本不足」。E 不是裁判能力或準確率評分。<span id="previewNote"></span></div></div>
__STATIC_MOBILE_FALLBACK__
<section class="panel"><h2>使用說明</h2><div class="cols3">
<div class="introitem"><strong>① 好球率是「比例」</strong><p>所有邊緣球好球率＝主審判好球數／所有邊緣球數。排行榜也能專看「理論壞球卻被判好球」的比例，分母只計理論壞球。</p></div>
<div class="introitem"><strong>② 比較數字不是判決正確率</strong><p>球卡上的 +／− pp，是扣除球點與球數基準後，與同區域其他主審的歷史差異。暖色較常判好球、冷色較少。</p></div>
<div class="introitem"><strong>③ E 只提醒樣本不足</strong><p>不顯示 A–D，也不以跨期正負變化當作主審品質評分。看球數、場次和 CI；「E」只代表這組資料尚不足以支持可靠比較。</p></div></div>
<details class="explain"><summary>展開統計與好球帶名詞說明</summary><dl class="glossary">
<dt>資料座標方向</dt><dd>Left / Right 為資料來源座標，尚未確認對應打者視角的內外角；並非內角／外角。</dd>
<dt>好球帶幾何</dt><dd>第三方資料的理論判決公式：矩形核心水平半寬 22.00 cm，圓角向外擴張 3.69 cm，最外側 ±25.69 cm；不等同官方裁判規則。</dd>
<dt>球的物理半徑</dt><dd>2024 年量測球徑 2.89 英吋，半徑約 3.6703 cm；不是來源演算法的 3.69 cm。</dd>
<dt>邊緣球</dt><dd>本網站使用距第三方理論好球帶邊界 10 公分內的投球，避免大量明顯好球或壞球稀釋差異。</dd>
<dt>「撿壞球」比率</dt><dd>分母：該區域且屬來源理論壞球的邊緣球。分子：其中被主審判為 STRIKE 的球。這是相對第三方幾何的判決分歧，不等於官方認定誤判。</dd>
<dt>漏判好球率</dt><dd>分母：該區域來源理論好球的邊緣球。分子：其中被主審判為 BALL 的球。</dd>
<dt>百分點 pp</dt><dd>兩個百分比相減，例如 55%−45%＝10 個百分點。</dd>
<dt>95% CI</dt><dd>針對主審歷史差異，按整場比賽重抽樣取得的區間；仍不包括所有模型估計不確定性，不能當成下一球預測範圍。排行榜不顯示 CI，只呈現實際比例及樣本警示。</dd>
<dt>跨期結果</dt><dd>前後期正負號或幅度可變，尤其接近零時不應過度解讀。因此移除舊版 D 評級，但保留前後期數字供參考。</dd>
<dt>模型驗證</dt><dd>Model A＝位置＋球數；B＝A＋主審身分；C＝B＋主審×位置交互項。使用日期切分測試。</dd>
<dt>歷史比較限制</dt><dd>不同主審所遇的投球條件並非完全相同；排行榜是未校正的觀察比例，不是因果效果或誰故意偏袒的證據。比較卡的殘差則有以無主審基準模型調整位置與球數。</dd>
</dl></details></section>
<section class="panel"><h2>資料來源與模型驗證</h2><div class="metricrow" id="globalMetrics"></div><p class="small" id="ballNote"></p><div class="splitline"></div><h3>Model A / B / C：時間樣本外預測</h3><div class="tablewrap"><table class="model"><thead><tr><th>模型</th><th>訓練 Log Loss</th><th>測試 Log Loss ↓</th><th>測試 Brier ↓</th><th>測試 AUC ↑</th></tr></thead><tbody id="modelRows"></tbody></table></div><p class="small" id="modelExplanation"></p></section>
<section class="panel"><h2>選擇兩位主審，對照歷史情蒐卡</h2><p class="small">先設定球數（預設 0 壞 0 好），再選主審。球數切換後五區數據與樣本資訊同步更新；全季涵蓋警示固定由全季資料決定。</p>
<div class="countbox"><div><h3>查看不同球數的主審反應</h3><p class="small">例如兩好兩壞：壞球數選 2、好球數選 2。只使用歷史上「當時正好是該球數」的投球；不是預測下一球。</p></div><div class="controls"><div><label for="balls">壞球數 Balls</label><select id="balls"><option value="0">0 壞</option><option value="1">1 壞</option><option value="2">2 壞</option><option value="3">3 壞</option></select></div><div><label for="strikes">好球數 Strikes</label><select id="strikes"><option value="0">0 好</option><option value="1">1 好</option><option value="2">2 好</option></select></div><button type="button" id="allCounts">查看所有球數的全季彙總</button></div><p id="countNote" class="countnote" role="status"></p><div class="dualhint"><b>樣本提示分成兩種：</b>① 全季歷史樣本是否足夠（未達門檻時才出現 ⚠ E）；② 目前球數的資料量（較充足／有限／不足／無資料）。<br>切換球數不會改動全季樣本警示。少於 10 球時，該球數的差異不展示。</div></div>
<section class="ranking"><div class="rankhead"><div><h2>聯盟主審判決排行榜</h2><p class="small">找出哪位主審歷史上最常在某一區域「撿理論壞球」。排行榜預設以全季所有球數排名，避免預設 0–0 過度切細導致大量 E；如需查看 2–2 等狀況，可改選「跟隨主審情蒐卡球數」。這是第三方理論好球帶下的觀察比例，不代表官方認定誤判或裁判能力。</p></div><span class="pill">排行榜預設全季球數</span></div>
<div class="controls">
<div><label for="rankScope">排行榜球數範圍</label><select id="rankScope"><option value="all">全季所有球數（推薦）</option><option value="follow">跟隨主審情蒐卡球數</option></select></div>
<div><label for="rankRegion">好球帶區域</label><select id="rankRegion"><option value="High">上緣 High</option><option value="Low">下緣 Low</option><option value="Left" selected>座標左側 Left</option><option value="Right">座標右側 Right</option><option value="Corner">角落 Corner</option></select></div>
<div><label for="rankMetric">排序指標</label><select id="rankMetric"><option value="False_Strike">理論壞球被判好球率（撿壞球）</option><option value="Called_Strike">所有邊緣球的實際好球率</option><option value="Missed_Strike">理論好球被判壞球率</option></select></div>
<div><label for="rankOrder">排序方式</label><select id="rankOrder"><option value="desc">由高到低</option><option value="asc">由低到高</option></select></div></div>
<div class="rankinfo" id="rankExplanation"></div><div class="tablewrap"><table class="leaderboard"><thead><tr><th>排名</th><th>主審</th><th>比例</th><th>符合條件球數／分母</th><th>場次</th><th>樣本提醒</th><th></th></tr></thead><tbody id="leaderboardRows"></tbody></table></div><p class="small" id="rankFootnote"></p>
</section>
<div class="controls"><div><label for="u1">主審 A</label><select id="u1"></select></div><div><label for="u2">主審 B</label><select id="u2"></select></div><div><label for="focus">聚焦區域</label><select id="focus"></select></div><button type="button" id="swap">交換主審</button><button type="button" id="print">列印／存 PDF</button></div><div class="cards"><article class="ump-card" id="card1"></article><article class="ump-card" id="card2"></article></div></section>
<section class="comparison"><h2 id="compareTitle">相同區域的主審相對偏好</h2><p class="small">圓點＝歷史估計值；有足夠投球時才顯示橫線＝95% Bootstrap CI；中間虛線 0＝相對其他主審沒有差異。</p><svg class="chart" id="chart" viewBox="0 0 980 245" aria-label="兩位主審的相對偏好與信賴區間" role="img"></svg><p class="small">提醒：兩位主審的估計值分別是「本人 vs 排除本人的其他主審」，不能直接將兩個數值相減並宣稱是嚴格的兩人配對效果。</p></section>
<footer>研究用途：第三方資料重建、歷史判決傾向分析。沒有主審能力或正確率分數；僅在樣本不足時顯示 E 警示。指定球數另顯示樣本量。不同好球帶位置、球數與比賽條件可能影響排行榜觀察比例；請勿視為主審的判決意圖或下一場預測。</footer>
</main>
<!-- 全部查詢資料直接內嵌於此 HTML，不需要外部 CSV、伺服器或網路請求。 -->
<script id="embedded-umpire-data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('embedded-umpire-data').textContent);

const byU={};for(const r of data.regions){(byU[r.Umpire]??={})[r.Region]=r;}
const over=Object.fromEntries(data.overall.map(r=>[r.Umpire,r]));
const byCount={};for(const r of data.count_regions){((byCount[r.Count]??={})[r.Umpire]??={})[r.Region]=r;}
const overCount={};for(const r of data.count_overall){(overCount[r.Count]??={})[r.Umpire]=r;}
const countSizes=Object.fromEntries(data.count_sizes.map(r=>[r.Count,r]));
const names=Object.keys(byU).sort((a,b)=>a.localeCompare(b,'zh-Hant'));
const REG=['High','Low','Left','Right','Corner'];
const LABEL={High:'上緣 High',Low:'下緣 Low',Left:'座標左側 Left',Right:'座標右側 Right',Corner:'角落／方向模糊'};
// All A-D evidence labels removed. Only sample coverage warning E remains.
const $=id=>document.getElementById(id);
const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const isNum=n=>n!==null&&n!==undefined&&n!==''&&Number.isFinite(Number(n));
const num=n=>!isNum(n)?'—':(Number(n)>0?'+':'')+Number(n).toFixed(1);
const fx=(n,d=3)=>!isNum(n)?'—':Number(n).toFixed(d);
const percent=n=>(Number(n)*100).toFixed(2)+'%';
// Full-season E warning ALWAYS comes from all-count byU, never count-conditioned results.
const seasonGrade=(name,region)=>{
  const r=byU[name]?.[region];
  return !r?'全季：無資料':r.Sample_Warning_E?'⚠ E｜全季樣本不足':'全季樣本符合門檻';
};
const seasonCode=(name,region)=>byU[name]?.[region]?.Sample_Warning_E?'E':'ok';
const sampleStatus=r=>{
  if(mode==='all')return '全季彙總';
  if(!r||!Number(r.N))return '無資料';
  // Keep UI synchronized with the status computed in Python.
  if(['較充足','有限','不足'].includes(r.Count_Sample_Status))return r.Count_Sample_Status;
  // Backward-compatible fallback for older exported count tables.
  if(Number(r.N)>=35 && Number(r.Games)>=6 &&
     Number(r.N_First_Half)>=10 && Number(r.N_Second_Half)>=10 &&
     Number(r.Games_First_Half)>=2 && Number(r.Games_Second_Half)>=2)return '較充足';
  if(Number(r.N)>=20 && Number(r.Games)>=4)return '有限';
  return '不足';
};
const currentSampleText=r=>mode==='all'?'目前球數樣本：全季彙總':`目前球數樣本：${sampleStatus(r)}`;
const nameOptions=id=>{const e=$(id);for(const n of names){const o=document.createElement('option');o.textContent=n;o.value=n;e.appendChild(o);}};
nameOptions('u1');nameOptions('u2');for(const reg of REG){const e=document.createElement('option');e.value=reg;e.textContent=LABEL[reg];$('focus').appendChild(e);}
$('u1').value=names.includes('陳均瑋')?'陳均瑋':names[0];$('u2').value=names.includes('吳家維')?'吳家維':(names[1]||names[0]);$('focus').value='Right';
$('balls').value='0';$('strikes').value='0';
let mode=data.count_regions.length?'selected':'all';
const countKey=()=>`${$('balls').value}-${$('strikes').value}`;
const modeName=()=>mode==='all'?'所有球數（全季彙總）':`${$('balls').value} 壞 ${$('strikes').value} 好`;
const selectedRegions=name=>mode==='all'?(byU[name]||{}):(byCount[countKey()]?.[name]||{});
const selectedOverall=name=>mode==='all'?(over[name]||{}):(overCount[countKey()]?.[name]||{});
const canDisplay=r=>r&&(mode==='all'||r.N>=10);
const g=data.geometry,c=data.counts;
$('sampleChip').textContent=`${c.umpires} 位主審 · ${c.pitches.toLocaleString('zh-TW')} 顆球 · ${c.games} 場`;
$('globalMetrics').innerHTML=`<div class="metric"><b>${percent(g.theoretical_call_agreement)}</b><span>理論好球帶重建吻合率</span></div><div class="metric"><b>${fx(g.distance_mae_cm,10)}</b><span>邊界距離平均誤差（cm）</span></div><div class="metric"><b>${c.edge.toLocaleString('zh-TW')}</b><span>全季邊緣球（≤ 10 cm）</span></div><div class="metric"><b>${c.umpires}</b><span>歷史主審人數</span></div>`;
$('ballNote').textContent=`2024 年中職用球直徑 2.89 英吋，物理半徑約 ${(g.physical_ball_radius_m_2024_reference_only*100).toFixed(4)} cm；資料來源的幾何圓角半徑 ${(g.source_rounded_expansion_m*100).toFixed(2)} cm。兩者不是同一個定義。`;
$('modelRows').innerHTML=data.models.map(r=>`<tr><td>${esc(r.Model)}</td><td>${fx(r.Train_LogLoss,5)}</td><td>${fx(r.Test_LogLoss,5)}</td><td>${fx(r.Test_Brier,5)}</td><td>${fx(r.Test_AUC,5)}</td></tr>`).join('');
const aa=data.models.find(r=>r.Model==='A_No_Umpire'),bb=data.models.find(r=>r.Model==='B_With_Umpire');
$('modelExplanation').textContent=aa&&bb?`樣本外 Model B 相較 A，Log Loss 改善 ${((aa.Test_LogLoss-bb.Test_LogLoss)/aa.Test_LogLoss*100).toFixed(2)}%。這是預測機率指標的相對改善，並非「主審判決準確率改善」。Model C 較複雜，必須觀察測試集結果避免過擬合。`:'模型結果請見上表';
$('previewNote').textContent=data.preview_label||'';
function updateCountNote(){
  const note=$('countNote');note.classList.remove('warning');
  if(!data.count_regions.length){note.textContent='目前是舊版分析的版面預覽，沒有原始球數明細。請將 umpire_pitch_calls.csv 放入完整程式套件，重新執行後才可使用球數切換。';note.classList.add('warning');$('balls').disabled=true;$('strikes').disabled=true;$('allCounts').disabled=true;return;}
  if(mode==='all'){note.textContent='目前查看所有球數彙總；所有比例與樣本數均使用全季資料。僅對樣本不足者顯示 E 警示。';return;}
  const k=countKey(),size=countSizes[k];
  note.textContent=`目前：${modeName()}。歷史上此球數共有 ${size?.Pitches??0} 顆邊緣球，涵蓋 ${size?.Games??0} 場。全季 E 樣本警示不隨球數改變；當前球數的樣本充分程度獨立更新。這是歷史觀察，不是下一球預測。`;
}
function cell(name,r,reg){
  const valid=canDisplay(r)&&isNum(r.Difference_pp),v=valid?r.Difference_pp:null;
  const klass=valid?(v>0?'positive':v<0?'negative':''):'';
  const season=seasonGrade(name,reg),sam=currentSampleText(r);
  const hint=`${LABEL[reg]}｜${valid?num(v)+' pp':'該球數資料不足'}｜${season}｜${sam}。點選此區域`;
  return `<div class="zone ${reg.toLowerCase()} ${klass} ${$('focus').value===reg?'active':''}" data-region="${reg}" tabindex="0" role="button" title="${esc(hint)}"><span class="zn">${LABEL[reg]}</span><span class="val">${num(v)}</span><span class="n">pp · ${r?.N??0} 球</span><span class="evidence">${season}</span><span class="sample">${sam}</span></div>`;
}
function card(n){
  const by=selectedRegions(n),o=selectedOverall(n),reg=$('focus').value,chosen=by[reg];
  const rows=REG.map(z=>{
    const r=by[z];
    const ci=isNum(r?.CI_Low_pp)&&isNum(r?.CI_High_pp)?`[${num(r.CI_Low_pp)}, ${num(r.CI_High_pp)}]`:'—';
    return `<tr><td>${LABEL[z]}</td><td class="num">${canDisplay(r)?num(r.Difference_pp):'—'}</td><td class="num">${ci}</td><td class="num">${r?.N??0} / ${r?.Games??0}</td><td><span class="tag ${esc(seasonCode(n,z))}">${seasonGrade(n,z)}</span></td><td>${currentSampleText(r)}</td></tr>`;
  }).join('');
  let msg=!chosen?`${n} 在「${LABEL[reg]}」及「${modeName()}」的歷史邊緣球沒有可比較的資料。`:!canDisplay(chosen)?`${n} 在「${LABEL[reg]}」、${modeName()}只有 ${chosen.N} 顆歷史邊緣球。少於 10 顆，不顯示偏好差異，避免過度解讀。`:
    `${n} 在「${LABEL[reg]}」、${modeName()}的歷史投球中，相較同區域${mode==='all'?'':'、同球數'}其他主審，扣除模型預期後的好球判決差異${chosen.Difference_pp>=0?'較高':'較低'}約 ${Math.abs(chosen.Difference_pp).toFixed(1)} 個百分點。此為歷史觀察，不是判決能力評分。`;
  const period=chosen?`前期：${num(chosen.Difference_pp_First_Half)} pp（${chosen.N_First_Half??0} 球）／後期：${num(chosen.Difference_pp_Second_Half)} pp（${chosen.N_Second_Half??0} 球）。`:'此組沒有足夠歷史投球';
  const limitations=mode==='all'?'全季歷史差異有探索性信賴區間，未做正式因果判定。':'目前球數是描述性分析，樣本不足時不顯示 CI；沒有 A–D 評級。';
  return `<div class="cardhead"><div><div class="small">${data.preview_label.includes('合成')?'⚠ 合成測試數值 · ':''}2026 全季 · ${modeName()} · 歷史情蒐</div><div class="name">${esc(n)}</div>${data.preview_label.includes('合成')?'<div class="small" style="color:#ffd3a5;font-weight:750">真實姓名僅供選單示意；本卡數據皆非真實判決結果</div>':''}</div><div class="evidencepair"><span class="pill">${seasonGrade(n,reg)}</span><span class="pill countpill">${currentSampleText(chosen)}</span></div></div><p class="summaryline">${esc(msg)}</p><div class="metricrow"><div class="metric"><b>${o.N??0}</b><span>${modeName()}邊緣球數</span></div><div class="metric"><b>${o.Games??0}</b><span>涵蓋場次</span></div><div class="metric"><b>${isNum(o.Actual)?fx(o.Actual,1)+'%':'—'}</b><span>實際好球率</span></div><div class="metric"><b>${isNum(o.Residual)?num(o.Residual)+' pp':'—'}</b><span>實際－預期（非同區域比較）</span></div></div><h3 class="subhead">五區判決傾向｜${modeName()}（全季僅在樣本不足時標 E）</h3><div class="zonegrid">${REG.map(z=>cell(n,by[z],z)).join('')}</div><div class="tablewrap"><table class="cardtable"><thead><tr><th>區域</th><th>當前差異 pp</th><th>當前 95% CI</th><th>當前球／場</th><th>全季樣本涵蓋提示（固定）</th><th>目前球數樣本充分程度（變動）</th></tr></thead><tbody>${rows}</tbody></table></div><div class="splitline"></div><h3>時間穩定性（${modeName()}）</h3><p class="small">${esc(period)}<br>${limitations}所有數值僅為歷史觀察，不是主審能力或下一球的預測。</p>`;
}
function graph(){
  const n1=$('u1').value,n2=$('u2').value,rg=$('focus').value,rs=[selectedRegions(n1)[rg],selectedRegions(n2)[rg]];
  const arr=rs.flatMap(r=>[r?.Difference_pp,r?.CI_Low_pp,r?.CI_High_pp]).filter(isNum).map(Number);
  const l=Math.min(-20,...arr)-3,h=Math.max(20,...arr)+3,x=v=>180+(v-l)/(h-l)*630;
  let s='<rect width="980" height="245" fill="transparent"/>';
  for(const t of [-20,-10,0,10,20]){const xx=x(t);s+=`<line x1="${xx}" y1="18" x2="${xx}" y2="206" stroke="${t===0?'#a3bacb':'#41556c'}" stroke-dasharray="${t===0?'5 4':'2 5'}"/><text x="${xx}" y="225" fill="#c4d1e0" text-anchor="middle" font-size="13">${t}</text>`;}
  rs.forEach((r,i)=>{const y=77+i*89,color=i?'#81baf1':'#f3a080',name=i?n2:n1;
    s+=`<text x="10" y="${y+5}" fill="#f6f9ff" font-size="18">${esc(name)}</text>`;
    if(!canDisplay(r)||!isNum(r.Difference_pp)){s+=`<text x="455" y="${y+5}" fill="#becbd8" font-size="16">${r?'少於 10 球：不展示差異':'此球數無可比較的資料'}</text>`;return;}
    if(isNum(r.CI_Low_pp)&&isNum(r.CI_High_pp)){
       s+=`<line x1="${x(r.CI_Low_pp)}" x2="${x(r.CI_High_pp)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="5"/><line x1="${x(r.CI_Low_pp)}" x2="${x(r.CI_Low_pp)}" y1="${y-8}" y2="${y+8}" stroke="${color}" stroke-width="3"/><line x1="${x(r.CI_High_pp)}" x2="${x(r.CI_High_pp)}" y1="${y-8}" y2="${y+8}" stroke="${color}" stroke-width="3"/>`;
    }else{s+=`<text x="465" y="${y-15}" fill="#ecc9a0" font-size="12">樣本少：不提供 CI</text>`;}
    s+=`<circle cx="${x(r.Difference_pp)}" cy="${y}" r="8" fill="${color}"/><text x="820" y="${y+6}" fill="${color}" font-size="18" font-weight="bold">${num(r.Difference_pp)} pp</text>`;
  });
  $('chart').innerHTML=s;
  $('compareTitle').textContent=`${LABEL[rg]}｜${modeName()}：兩位主審分別相對其他主審的歷史差異`;
}

const league=data.rankings||[];
const rankIndex={};for(const r of league){((rankIndex[r.Count]??={})[r.Region]??=[]).push(r);}
const rankingMetricDef={
  False_Strike:{rate:'False_Strike_Rate_pct',numerator:'False_Strike_N',denominator:'Theoretical_Ball_N',games:'Theoretical_Ball_Games',eligible:'False_Strike_Eligible',name:'理論壞球被判好球率',denomText:'同區域的第三方理論壞球'},
  Called_Strike:{rate:'Called_Strike_Rate_pct',numerator:'Called_Strike_N',denominator:'N_Edge',games:'Games',eligible:'Called_Strike_Eligible',name:'所有邊緣球的實際好球率',denomText:'同區域的全部邊緣球'},
  Missed_Strike:{rate:'Missed_Strike_Rate_pct',numerator:'Missed_Strike_N',denominator:'Theoretical_Strike_N',games:'Theoretical_Strike_Games',eligible:'Missed_Strike_Eligible',name:'理論好球被判壞球率',denomText:'同區域的第三方理論好球'}
};
function renderLeaderboard(){
  const rg=$('rankRegion').value,metric=$('rankMetric').value, order=$('rankOrder').value;
  const def=rankingMetricDef[metric], state=($('rankScope').value==='all'||mode==='all')?'ALL':countKey();
  const rankScopeName=state==='ALL'?'全季所有球數':modeName();
  const rows=[...(rankIndex[state]?.[rg]||[])];
  // Ranking only among cells that meet the relevant denominator + game threshold;
  // E cells are shown underneath, WITHOUT an official rank number.
  rows.sort((a,b)=>{
    const ea=Boolean(a[def.eligible]),eb=Boolean(b[def.eligible]);if(ea!==eb)return ea?-1:1;
    const va=isNum(a[def.rate])?Number(a[def.rate]):(order==='desc'?-Infinity:Infinity);
    const vb=isNum(b[def.rate])?Number(b[def.rate]):(order==='desc'?-Infinity:Infinity);
    if(va!==vb)return order==='desc'?vb-va:va-vb;
    return a.Umpire.localeCompare(b.Umpire,'zh-Hant');
  });
  const nEligible=rows.filter(r=>r[def.eligible]).length;
  const label=LABEL[rg];
  $('rankExplanation').textContent=`${rankScopeName}｜${label}｜${def.name}：分子為符合條件的實際判決數，分母為「${def.denomText}」。目前 ${nEligible} 位主審達排行門檻（至少 ${metric==='Called_Strike'?'30 球／5 場':'20 球／4 場'}）；其餘會標示 E 並排在後面。`;
  $('leaderboardRows').innerHTML=rows.map((r,i)=>{
    const eligible=Boolean(r[def.eligible]),v=r[def.rate],good=isNum(v);
    const rank=eligible?String(rows.slice(0,i+1).filter(t=>t[def.eligible]).length):'—';
    return `<tr class="${eligible?'':'warnrow'}"><td>${rank}</td><td>${esc(r.Umpire)}</td><td class="value ${eligible?'strong':''}">${good?fx(v,1)+'%':'—'}</td><td>${r[def.numerator]??0}／${r[def.denominator]??0}</td><td>${r[def.games]??0}</td><td>${eligible?'樣本達門檻':'<span class="rankwarn">⚠ E｜樣本不足</span>'}</td><td><button type="button" class="rank-view" data-name="${esc(r.Umpire)}">查看情蒐卡</button></td></tr>`;
  }).join('')||'<tr><td colspan="7">此球數／區域沒有歷史資料。</td></tr>';
  $('rankFootnote').textContent='僅按來源理論好球帶與實際判決計算的歷史比例。E 項目不參與正式排名。未校正不同主審的球點分布、球員或其他比賽情境；不代表裁判故意誤判。';
  $('leaderboardRows').querySelectorAll('.rank-view').forEach(b=>b.addEventListener('click',()=>{
    $('u1').value=b.dataset.name;
    $('focus').value=rg;
    render();
    $('card1').scrollIntoView({behavior:'smooth',block:'start'});
  }));
}
for(const id of ['rankScope','rankRegion','rankMetric','rankOrder'])$(id).addEventListener('change',renderLeaderboard);
function render(){updateCountNote();renderLeaderboard();$('card1').innerHTML=card($('u1').value);$('card2').innerHTML=card($('u2').value);graph();document.querySelectorAll('.zone').forEach(el=>{const f=()=>{$('focus').value=el.dataset.region;render();};el.addEventListener('click',f);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();f();}});});}
for(const id of ['u1','u2','focus'])$(id).addEventListener('change',render);
for(const id of ['balls','strikes'])$(id).addEventListener('change',()=>{mode='selected';render();});
$('allCounts').addEventListener('click',()=>{mode='all';render();});
$('swap').addEventListener('click',()=>{const a=$('u1').value;$('u1').value=$('u2').value;$('u2').value=a;render();});
$('print').addEventListener('click',()=>window.print());
render();
// The static snapshot stays visible in restricted viewers or if JS fails.
document.getElementById('staticMobileFallback')?.remove();
</script></body></html>'''
    html_doc=template.replace('__DATA__',payload).replace('__STATIC_MOBILE_FALLBACK__',static_fallback)
    out.write_text(html_doc,encoding='utf-8')
    static_path=out.with_name('umpire_scouting_mobile_static.html')
    static_path.write_text(standalone_mobile_html(body), encoding='utf-8')
    print(f'[手機純靜態版] {static_path}')
    print(f'[HTML] {out} ({len(html_doc.encode("utf-8")):,} bytes；內嵌統計資料與靜態備援，無外部資料依賴)')


def run_pipeline(args):
    inp=Path(args.input).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    d=load_csv(inp)
    inferred=infer_geometry_from_source(d)
    (out/'geometry_inference.json').write_text(json.dumps(inferred,ensure_ascii=False,indent=2),encoding='utf-8')
    d,geom=add_geometry(d)
    (out/'geometry_validation.json').write_text(json.dumps(geom,ensure_ascii=False,indent=2),encoding='utf-8')
    model,boot=model_comparison(d,pd.Timestamp(args.split),out,args.bootstrap)
    region,overall,edge_count,mid_date,historical_edge=relative_historical(d,out,args.bootstrap)
    count_regions,count_overall,count_sizes=count_conditioned_historical(historical_edge,out,args.bootstrap)
    rankings=league_rankings(d,out)
    write_html(geom,model,boot,region,overall,len(d),d.game_id.nunique(),d.umpire_name.nunique(),edge_count,out/'umpire_scouting.html',
               count_regions=count_regions,count_overall=count_overall,count_sizes=count_sizes,rankings=rankings,
               preview_label=('⚠ 合成測試數據：主審姓名僅供介面示範；所有數值與排名均非真實中職結果。' if args.synthetic_demo else ''))
    print(f'\n全部完成！請開啟：{out/"umpire_scouting.html"}')


def main():
    ap=argparse.ArgumentParser(description='中職主審研究：從原始逐球 CSV 一鍵產生離線互動式歷史情蒐頁面')
    ap.add_argument('--input',default='umpire_pitch_calls.csv',help='原始逐球 CSV')
    ap.add_argument('--out',default='output',help='輸出資料夾')
    ap.add_argument('--split',default='2026-09-16',help='樣本外測試起始日期 YYYY-MM-DD')
    ap.add_argument('--bootstrap',type=int,default=4000,help='按比賽重抽樣次數')
    ap.add_argument('--synthetic-demo',action='store_true',help='僅供開發者使用：將合成資料輸出加上醒目警示')
    args=ap.parse_args()
    if args.bootstrap<200:raise ValueError('Bootstrap 至少 200 次')
    run_pipeline(args)


if __name__=='__main__':main()
