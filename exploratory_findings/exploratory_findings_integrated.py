#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable counterpart of exploratory_findings_integrated.ipynb.
Run from any directory: python /path/to/exploratory_findings/exploratory_findings_integrated.py
The paired Jupyter notebook explains each result and its caveats.
"""

# %%
from pathlib import Path
import os, sys, json, platform, warnings, codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
import statsmodels.formula.api as smf
from IPython.display import display, Markdown

# Shared study inputs remain in the parent project; the exposure audit is local.
# Jupyter's working directory can be unavailable (e.g., a moved folder or OS permissions).
# Resolve known absolute locations first, and never require Path.cwd() to succeed.
# Optional, when automatic detection fails:
#     os.environ['EXPLORATORY_FINDINGS_DIR'] = '/absolute/path/to/exploratory_findings'
def resolve_analysis_directory():
    candidates = []
    seen = set()

    def add_anchor(anchor):
        if not anchor:
            return
        try:
            path = Path(anchor).expanduser()
            # Resolving a relative path may itself call os.getcwd(), which can fail.
            if not path.is_absolute():
                return
            options = (
                path,
                path / 'exploratory_findings',
                path.parent / 'exploratory_findings',
                path / 'historical-ai-interfaces' / 'exploratory_findings',
                path / 'avatar_vs_chat' / 'exploratory_findings',
            )
            for candidate in options:
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
        except (OSError, RuntimeError, ValueError):
            return

    # The explicit setting must be consulted before anything that can call getcwd().
    add_anchor(os.environ.get('EXPLORATORY_FINDINGS_DIR'))
    if '__file__' in globals():
        script_path = Path(__file__).expanduser()
        if script_path.is_absolute():
            add_anchor(script_path.parent)

    # PWD may remain usable even if Python's os.getcwd() is denied.
    for name in ('PWD', 'OLDPWD', 'JUPYTER_SERVER_ROOT'):
        add_anchor(os.environ.get(name))

    # Notebook kernels may put the notebook directory on sys.path.
    for entry in sys.path:
        if isinstance(entry, str) and entry:
            add_anchor(entry)

    try:
        add_anchor(Path.cwd())
    except OSError:
        # Do not fail merely because Jupyter's current directory is inaccessible.
        pass

    for candidate in candidates:
        try:
            parent = candidate.parent
            if ((candidate / 'exploratory_findings_integrated.py').is_file()
                    and (parent / 'data' / 'study_data.csv').is_file()
                    and (parent / 'analysis_output' / 'analysis_merged_interaction_level.csv').is_file()):
                return candidate
        except OSError:
            continue

    raise FileNotFoundError(
        'Could not locate exploratory_findings and its parent study data. '
        'If the Jupyter working directory is inaccessible, restart the kernel '
        'from a readable project folder. Alternatively, put this line at the '
        'start of the first notebook code cell (before resolve_analysis_directory):\n'
        "os.environ['EXPLORATORY_FINDINGS_DIR'] = '/absolute/path/to/exploratory_findings'\n"
        'Then run the notebook again.'
    )

ROOT = resolve_analysis_directory()
PROJECT_ROOT = ROOT.parent
INPUT_FILES = {
    'interactions': PROJECT_ROOT / 'analysis_output' / 'analysis_merged_interaction_level.csv',
    'study': PROJECT_ROOT / 'data' / 'study_data.csv',
    'messages': PROJECT_ROOT / 'data' / 'coded-messages-three-models.csv',
    'knowledge': PROJECT_ROOT / 'analysis_output' / 'knowledge_items.csv',
    'exposure': ROOT / 'data' / 'exposure_quiz_items_JOINED.csv',
}
missing = [str(path) for path in INPUT_FILES.values() if not path.is_file()]
if missing:
    raise FileNotFoundError('Required input files are missing:\n' + '\n'.join(missing))
# Switch to a known accessible directory. Some notebook kernels retain a stale
# working directory after a project is moved, which can break Python imports.
try:
    os.chdir(ROOT)
except OSError as exc:
    raise PermissionError(
        f'Cannot use the analysis directory as the working directory: {ROOT}. '
        'Check folder permissions and the EXPLORATORY_FINDINGS_DIR setting.'
    ) from exc

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True, parents=True)
print('Project root:', PROJECT_ROOT)
print('Analysis folder:', ROOT)
print('Results:', OUT)
print('Python:', sys.version.split()[0], '| pandas:', pd.__version__)

# %%
def read_study_csv(path):
    """Read UTF-8 CSV without pandas' optional zstandard importer.

    In some Jupyter kernels, pandas' optional-import probe scans a stale or
    restricted Python search path and raises PermissionError. A codecs
    StreamReader is a text reader recognized by pandas, so it avoids the
    unrelated optional-import probe without modifying pandas internals.
    """
    try:
        with Path(path).open('rb') as binary_stream:
            text_stream = codecs.getreader('utf-8-sig')(binary_stream)
            return pd.read_csv(text_stream, low_memory=False)
    except PermissionError as exc:
        raise PermissionError(
            f'Cannot open required input file: {path}. '
            'Check the file permissions or move the project to a readable folder.'
        ) from exc


d = read_study_csv(INPUT_FILES['interactions'])
raw = read_study_csv(INPUT_FILES['study'])
msg = read_study_csv(INPUT_FILES['messages'])
quiz = read_study_csv(INPUT_FILES['knowledge'])
exp = read_study_csv(INPUT_FILES['exposure'])
assert len(d)==292 and d.user_id.nunique()==146
assert (d.groupby('user_id').size()==2).all()
assert (d.groupby('user_id').condition.nunique()==2).all()
assert set(d.condition)=={'Avatar','Chat'}
assert len(quiz)==1460 and len(quiz.groupby(['user_id','stage_id']))==1460
assert len(msg)>2000
study_coverage = {'original_participants':int(d.user_id.nunique()), 'original_interactions':len(d),
     'quiz_items':len(quiz), 'coded_user_messages':len(msg),
     'LLM_exposure_items':len(exp), 'LLM_audited_interactions':int(exp[['user_id','condition']].drop_duplicates().shape[0]) if not exp.empty else 0}
print(json.dumps(study_coverage, ensure_ascii=False, indent=2))
assert exp.empty or (len(exp)==1445 and exp[['user_id','condition']].drop_duplicates().shape[0]==289)
pd.DataFrame([study_coverage]).to_csv(OUT/'sample_coverage.csv', index=False)

# %%
def paired(df, outcome, group, family):
    w=df.pivot(index='user_id', columns='condition', values=outcome)
    w=w[['Avatar','Chat']].apply(pd.to_numeric, errors='coerce').dropna()
    v=(w['Avatar']-w['Chat']).astype(float)
    n=len(v)
    if n < 3: raise ValueError(f'Too few paired rows for {group} {outcome}')
    sd=float(v.std(ddof=1)); sem=sd/np.sqrt(n)
    ci=stats.t.interval(.95,n-1,loc=float(v.mean()),scale=sem) if sem>0 else (float(v.mean()),)*2
    if (v!=0).any():
        test=stats.wilcoxon(v,alternative='two-sided',zero_method='wilcox',method='auto')
        p=float(test.pvalue); stat=float(test.statistic)
    else:
        p=1.0; stat=0.0
    return dict(family=family,group=str(group),outcome=outcome,n_pairs=n,
                avatar_mean=float(w.Avatar.mean()),chat_mean=float(w.Chat.mean()),
                mean_diff=float(v.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                sd_diff=sd,cohens_dz=(float(v.mean()/sd) if sd>0 else np.nan),
                wilcoxon_stat=stat,p_raw=p,
                avatar_higher=int((v>0).sum()),chat_higher=int((v<0).sum()),ties=int((v==0).sum()))

def holm_family(table):
    o=table.copy()
    o['p_holm_family']=o.groupby('family')['p_raw'].transform(lambda p:multipletests(p,method='holm')[1])
    return o

def show_save(df, filename, cols=None):
    df.to_csv(OUT/filename,index=False)
    display(df[cols] if cols else df)
    return df

results=[] # all searched pairwise Wilcoxon contrasts recorded for the combined exploratory sensitivity correction

def add_rows(frame):
    results.extend(frame.to_dict('records'))

wide=d.pivot(index='user_id',columns='condition')
meta=d.drop_duplicates('user_id').set_index('user_id')
assert len(wide)==146

# Test whether the main inference changes with the study assignment and country.
def moderation_test(df, outcome, moderator):
    x=df.pivot(index='user_id',columns='condition',values=outcome)
    x=(x.Avatar-x.Chat).rename('paired_delta').to_frame()
    m=df[['user_id',moderator]].drop_duplicates('user_id').set_index('user_id')
    z=x.join(m).dropna()
    fit=smf.ols(f'paired_delta ~ C({moderator})',data=z.reset_index()).fit(cov_type='HC3')
    tab=fit.wald_test_terms(scalar=True).table
    return {'moderator':moderator,'outcome':outcome,'n':len(z),
            'robust_omnibus_p':float(tab.loc[f'C({moderator})','pvalue'])}

# %%
primary=holm_family(pd.DataFrame([paired(d,v,'all_146','primary_3') for v in
                                ('character_connection','satisfaction','knowledge_accuracy')]))
show_save(primary,'01_primary_paired.csv', ['outcome','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','cohens_dz','p_raw','p_holm_family'])
# The 146 knowledge scores are 5-item immediate recall, not change from baseline.
v=wide['used_voice'][['Avatar','Chat']].astype(bool)
a=v['Avatar'];c=v['Chat']
voice_cells={'n00_neither':int((~a&~c).sum()),'n01_chat_only':int((~a&c).sum()),
             'n10_avatar_only':int((a&~c).sum()),'n11_both':int((a&c).sum())}
voice_p=float(mcnemar([[voice_cells['n00_neither'],voice_cells['n01_chat_only']],
                      [voice_cells['n10_avatar_only'],voice_cells['n11_both']]],exact=True).pvalue)
voice_context=pd.DataFrame([{'n_participants':146,**voice_cells,'mcnemar_exact_p':voice_p}])
show_save(voice_context,'01_voice_mcnemar_context.csv')

# %%
pref=[]
for group,sub in d.groupby('explanation_type_preference'):
    for v in ['character_connection','satisfaction']:
        pref.append(paired(sub,v,group,'preference_6'))
pref=holm_family(pd.DataFrame(pref));add_rows(pref)
show_save(pref,'02_preference_subgroups.csv',['group','outcome','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family'])
pref_mod=pd.DataFrame([moderation_test(d,v,'explanation_type_preference') for v in ['character_connection','satisfaction']])
pref_mod['p_holm_2']=multipletests(pref_mod.robust_omnibus_p,method='holm')[1]
show_save(pref_mod,'02_preference_moderation.csv')
print('Do not infer greater Avatar effects in chat_only than in other groups from within-group p values.')

# %%
never_voice_ids=wide.index[(~wide['used_voice']['Avatar'].astype(bool)) & (~wide['used_voice']['Chat'].astype(bool))]
typed=holm_family(pd.DataFrame([paired(d[d.user_id.isin(never_voice_ids)],v,'never_voice_in_either','typed_2') for v in ['character_connection','satisfaction']]))
add_rows(typed)
show_save(typed,'03_never_voice.csv',['outcome','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family'])

# %%
judge_cols=['gemini-flash__tone','gpt-54__tone','claude-sonnet-5__tone']
assert msg[judge_cols].notna().all().all()
emo=msg[['user_id','condition']+judge_cols].copy()
emo['n_emotional']=(emo[judge_cols]=='emotional').sum(axis=1)
emo['majority_emotional']=(emo.n_emotional>=2).astype(int)
emo['any_judge_emotional']=(emo.n_emotional>=1).astype(int)
agg=emo.groupby(['user_id','condition'])[['majority_emotional','any_judge_emotional']].sum().unstack('condition',fill_value=0)
assert agg.shape[0]==146
none_majority=agg.index[(agg['majority_emotional']['Avatar']==0)&(agg['majority_emotional']['Chat']==0)]
none_any=agg.index[(agg['any_judge_emotional']['Avatar']==0)&(agg['any_judge_emotional']['Chat']==0)]
emot=[]
for name,ids in [('no_majority_emotional_either',none_majority),('no_any_judge_emotional_either',none_any)]:
    for y in ['character_connection','satisfaction']:
        emot.append(paired(d[d.user_id.isin(ids)],y,name,f'emotion_{name}_2'))
emot=holm_family(pd.DataFrame(emot));add_rows(emot)
show_save(emot,'04_emotionless_message_subgroups.csv',['group','outcome','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family'])
# Automated coder agreement on binary emotional/not emotional, for transparency.
def cohen_kappa_binary(a,b):
    a=np.asarray(a,dtype=int);b=np.asarray(b,dtype=int)
    observed=np.mean(a==b)
    expected=np.mean(a)*np.mean(b)+(1-np.mean(a))*(1-np.mean(b))
    return float((observed-expected)/(1-expected)) if expected<1 else np.nan
krows=[]
for a in range(3):
    for b in range(a+1,3):
        aa=(emo[judge_cols[a]]=='emotional').astype(int)
        bb=(emo[judge_cols[b]]=='emotional').astype(int)
        krows.append({'coder_a':judge_cols[a],'coder_b':judge_cols[b], 'n_messages':len(emo),
                      'cohens_kappa_emotional_binary':cohen_kappa_binary(aa,bb)})
show_save(pd.DataFrame(krows),'04_automatic_judge_agreement.csv')

# %%
first=d[d.condition_position==1][['user_id','condition']].rename(columns={'condition':'first_interface'})
d2=d.merge(first,on='user_id',how='left',validate='many_to_one')
assert first.first_interface.value_counts().to_dict()=={'Avatar':73,'Chat':73}
p=meta.copy()
p['first_interface']=first.set_index('user_id').first_interface
p['ai_experience_category']=p.ai_experience_years.astype(str)
times=pd.to_datetime(d.groupby('user_id').first_real_message_time.min(),utc=True,errors='coerce')
assert times.notna().all()
p['collection_period_proxy']=np.where(times<pd.Timestamp('2026-06-05',tz='UTC'),'before_2026_06_05','from_2026_06_05')
strata=[];moder=[]
specs=[('order','first_interface', ['Avatar','Chat']),('country','country', ['israel','germany','china']),
       ('AI_experience','ai_experience_category',['1-2','3-4','5']),
       ('period','collection_period_proxy',['before_2026_06_05','from_2026_06_05'])]
for family,field,groups in specs:
    dd=d.merge(p[[field]].reset_index(),on='user_id',how='left',validate='many_to_one',suffixes=('','_meta')) if field not in d.columns else d
    if field not in dd.columns: raise ValueError(field)
    for group in groups:
        ids=p.index[p[field]==group]
        for y in ['character_connection','satisfaction']:
            strata.append(paired(d[d.user_id.isin(ids)],y,group,'stratum_'+family))
    # Exclude the tiny 0-1 AI experience group from the stratified tests but include it for moderation.
    for y in ['character_connection','satisfaction']:
        mm=moderation_test(dd,y,field)
        mm['family']=family
        moder.append(mm)
strata=holm_family(pd.DataFrame(strata));add_rows(strata)
show_save(strata,'05_order_country_AI_period.csv',['family','group','outcome','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family'])
moder=pd.DataFrame(moder)
moder['p_holm_eight']=multipletests(moder.robust_omnibus_p,method='holm')[1]
show_save(moder,'05_moderation_robust_HC3.csv')
show_save(p.groupby(['collection_period_proxy','country']).size().rename('n').reset_index(),'05_period_country_confounding.csv')

# %%
ids=['satisfaction_q1','satisfaction_q2','satisfaction_q3','character_connection_q1','character_connection_q2']
scale=raw[raw.field_id.str.contains(r'(?:satisfaction_q[123]|character_connection_q[12])$',regex=True,na=False)].copy()
assert len(scale)==1460
scale['item_code']=scale.field_id.str.replace(r'^(?:ben_lesser|markel)_','',regex=True)
scale['character']=np.where(scale.field_id.str.startswith('ben_lesser_'),'Ben Lesser','Markel')
scale['numeric_value']=pd.to_numeric(scale.value,errors='coerce')
assert scale.numeric_value.notna().all()
scale=scale.merge(d[['user_id','session_id','character','condition']],on=['user_id','session_id','character'],how='inner',validate='many_to_one')
assert len(scale)==1460
itemtests=[]
for item in ids:
    z=scale[scale.item_code==item].rename(columns={'numeric_value':'score'})
    itemtests.append(paired(z,'score',item,'individual_scale_items_5'))
itemtests=holm_family(pd.DataFrame(itemtests));add_rows(itemtests)
show_save(itemtests,'06_five_questionnaire_items.csv',['group','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family'])

# %%
same_ids=wide.index[np.isclose(wide['knowledge_accuracy']['Avatar'],wide['knowledge_accuracy']['Chat'])]
same=holm_family(pd.DataFrame([paired(d[d.user_id.isin(same_ids)],v,'same_five_item_score_both','same_knowledge_2') for v in ['character_connection','satisfaction']]))
add_rows(same)
show_save(same,'07_same_knowledge_score.csv',['outcome','n_pairs','avatar_mean','chat_mean','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family'])

# %%
joint=[]
for th in [4.0,3.5,4.5]:
    a=(wide['character_connection']['Avatar']>=th)&(wide['satisfaction']['Avatar']>=th)
    c=(wide['character_connection']['Chat']>=th)&(wide['satisfaction']['Chat']>=th)
    n00=int((~a&~c).sum());n01=int((~a&c).sum());n10=int((a&~c).sum());n11=int((a&c).sum())
    diff=a.astype(int)-c.astype(int)
    # Cluster is the participant, already one matched pair per participant.
    ci=stats.t.interval(.95,len(diff)-1,loc=float(diff.mean()),scale=float(stats.sem(diff)))
    test=mcnemar([[n00,n01],[n10,n11]],exact=True)
    joint.append(dict(threshold=th,n_participants=len(diff),avatar_high=int(a.sum()),chat_high=int(c.sum()),
                      avatar_only=n10,chat_only=n01,both=n11,neither=n00,
                      paired_risk_difference=float(diff.mean()),approx_ci_low=float(ci[0]),approx_ci_high=float(ci[1]),p_raw=float(test.pvalue)))
joint=pd.DataFrame(joint)
joint['p_holm_3_thresholds']=multipletests(joint.p_raw,method='holm')[1]
show_save(joint,'08_joint_high_experience.csv')

# %%
def tost(outcome, margin):
    x=wide[outcome]['Avatar'].astype(float)-wide[outcome]['Chat'].astype(float)
    x=x.dropna();n=len(x);mean=float(x.mean());se=float(stats.sem(x))
    if se==0:raise ValueError('No variation for TOST')
    p_lower=float(stats.t.sf((mean+margin)/se,n-1))
    p_upper=float(stats.t.cdf((mean-margin)/se,n-1))
    ci90=stats.t.interval(.90,n-1,loc=mean,scale=se)
    return dict(outcome=outcome,n=n,margin=margin,mean_diff=mean,ci90_low=float(ci90[0]),ci90_high=float(ci90[1]),
                TOST_p=max(p_lower,p_upper),p_test_above_lower=p_lower,p_test_below_upper=p_upper,
                equivalent_at_alpha_05=bool(max(p_lower,p_upper)<.05))
eq=pd.DataFrame([tost('knowledge_accuracy',.10),tost('knowledge_accuracy',.05),
                  tost('n_messages',1.0),tost('n_messages',.5)])
show_save(eq,'09_tost_sensitivity.csv')

# %%
if not exp.empty:
    # This table contains the identified audited items after deterministic content-based matching.
    exp['is_correct']=pd.to_numeric(exp.is_correct)
    exp['exposed']=pd.to_numeric(exp.exposed)
    exp['conversation_sufficient']=pd.to_numeric(exp.conversation_sufficient)
    assert len(exp)==1445
    pair_ids=exp.groupby('user_id').condition.nunique()
    pair_ids=pair_ids[pair_ids==2].index
    xp=exp[exp.user_id.isin(pair_ids)].copy()
    assert len(pair_ids)==143 and len(xp)==1430
    desc=exp.groupby(['condition','exposed'],as_index=False).is_correct.agg(n='size',correct='sum',accuracy='mean')
    show_save(desc,'10_exposure_correct_descriptive.csv')
    exposure_tests=[]
    for e_var in ['exposed','conversation_sufficient']:
        formula=f'is_correct ~ {e_var}*C(condition) + C(stage_id_question)+C(country)+C(condition_position)'
        fit=smf.gee(formula,groups='user_id',data=xp,family=Binomial(),cov_struct=Exchangeable()).fit(maxiter=200)
        for term in [e_var,'C(condition)[T.Chat]',f'{e_var}:C(condition)[T.Chat]']:
            b=float(fit.params[term]);ci=fit.conf_int().loc[term]
            exposure_tests.append({'exposure_definition':e_var,'term':term,'n_items':len(xp),'n_participants':len(pair_ids),
                 'OR':float(np.exp(b)),'ci95_low':float(np.exp(ci.iloc[0])),'ci95_high':float(np.exp(ci.iloc[1])),
                 'p':float(fit.pvalues[term]),'formula':formula})
    gee=pd.DataFrame(exposure_tests)
    gee['p_holm_six_model_terms']=multipletests(gee.p,method='holm')[1]
    show_save(gee,'10_exposure_accuracy_GEE.csv')
    # Pairwise exposure coverage by condition, 143 exact pairs.
    cov=xp.groupby(['user_id','condition'],as_index=False).exposed.sum().rename(columns={'exposed':'n_exposed'})
    coverage=paired(cov,'n_exposed','paired_143','exposure_coverage')
    show_save(pd.DataFrame([coverage]),'10_exposure_paired_143.csv')
else:
    print('Exposure table is empty; exposure-specific analyses were skipped.')

# %%
items=quiz.merge(d[['user_id','session_id','character','condition']],
    on=['user_id','session_id','character'],how='inner',validate='many_to_one')
assert len(items)==1460
item_grid=[]
for (character,qnum),z in items.groupby(['character','question_number']):
    t=z.groupby('condition').is_correct.agg(n='size',correct='sum')
    av=int(t.loc['Avatar','correct']);an=int(t.loc['Avatar','n']);
    ch=int(t.loc['Chat','correct']);cn=int(t.loc['Chat','n'])
    odds,pval=stats.fisher_exact([[av,an-av],[ch,cn-ch]],alternative='two-sided')
    item_grid.append({'character':character,'question_number':int(qnum),'avatar_correct':av,
        'avatar_n':an,'chat_correct':ch,'chat_n':cn,'avatar_accuracy':av/an,'chat_accuracy':ch/cn,
        'risk_difference':av/an-ch/cn,'fisher_odds_ratio':odds,'p_raw':pval})
item_grid=pd.DataFrame(item_grid).sort_values(['character','question_number'])
assert len(item_grid)==10
item_grid['p_holm_ten']=multipletests(item_grid.p_raw,method='holm')[1]
show_save(item_grid,'10b_knowledge_item_fisher_Holm.csv')

# %%
delta=pd.DataFrame({
   'delta_connection':(wide['character_connection']['Avatar']-wide['character_connection']['Chat']).astype(float),
   'delta_satisfaction':(wide['satisfaction']['Avatar']-wide['satisfaction']['Chat']).astype(float),
   'delta_voice':(wide['voice_message_rate']['Avatar']-wide['voice_message_rate']['Chat']).astype(float),
   'country':meta['country']}).dropna()
pear=stats.pearsonr(delta.delta_connection,delta.delta_satisfaction)
spear=stats.spearmanr(delta.delta_connection,delta.delta_satisfaction)
model=smf.ols('delta_satisfaction ~ delta_connection + delta_voice + C(country)',data=delta.reset_index()).fit(cov_type='HC3')
ci=model.conf_int().loc['delta_connection']
summary_assoc=pd.DataFrame([{'n_pairs':len(delta),'pearson_r':pear.statistic,'pearson_p':pear.pvalue,
    'spearman_rho':spear.statistic,'spearman_p':spear.pvalue,'adjusted_B_connection':model.params['delta_connection'],
    'adjusted_B_ci95_low':ci.iloc[0],'adjusted_B_ci95_high':ci.iloc[1],
    'adjusted_B_p':model.pvalues['delta_connection'],'adjusted_voice_p':model.pvalues['delta_voice']}])
show_save(summary_assoc,'10c_connection_satisfaction_association.csv')

# %%
if not exp.empty:
    counts=exp.groupby(['user_id','condition'],as_index=False).exposed.sum().rename(columns={'exposed':'n_exposed'})
    adjusted=d.merge(counts,on=['user_id','condition'],how='inner',validate='one_to_one')
    complete=adjusted.groupby('user_id').condition.nunique(); adjusted=adjusted[adjusted.user_id.isin(complete[complete==2].index)]
    assert adjusted.user_id.nunique()==143
    rows=[]
    formulas=[('design_only','C(condition)+C(character)+C(country)+C(condition_position)'),
              ('plus_post_treatment_exposure_messages','C(condition)+C(character)+C(country)+C(condition_position)+n_exposed+n_messages')]
    for y in ['character_connection','satisfaction']:
        for name,pred in formulas:
            fit=smf.gee(f'{y} ~ {pred}',groups='user_id',data=adjusted,cov_struct=Exchangeable()).fit()
            term='C(condition)[T.Chat]'
            ci=fit.conf_int().loc[term]
            rows.append(dict(outcome=y,model=name,n_participants=143,avatar_minus_chat=-float(fit.params[term]),
                     ci95_low=-float(ci.iloc[1]),ci95_high=-float(ci.iloc[0]),p=float(fit.pvalues[term])))
    adjusted_table=pd.DataFrame(rows)
    adjusted_table['p_holm_4']=multipletests(adjusted_table.p,method='holm')[1]
    show_save(adjusted_table,'11_experience_adjusted_GEE.csv')

# %%
all_new=pd.DataFrame(results)
all_new['p_holm_across_notebook_exploratory']=multipletests(all_new.p_raw,method='holm')[1]
all_new['direction']='Avatar' 
all_new.loc[all_new.mean_diff<0,'direction']='Chat'
all_new.loc[all_new.mean_diff==0,'direction']='tied'
all_new=all_new.sort_values(['family','group','outcome']).reset_index(drop=True)
show_save(all_new,'12_all_exploratory_Wilcoxon_tests.csv',
  ['family','group','outcome','n_pairs','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family','p_holm_across_notebook_exploratory'])
print('Number of exploratory paired tests listed in notebook:',len(all_new))
print('Note: joint-threshold McNemar tests, item exposure GEE and TOST are separate families and NOT included above.')

# %%
plot_data=pd.concat([pref,typed,strata[strata.family.isin(['stratum_order','stratum_country'])]],ignore_index=True)
for y,name in [('character_connection','Connection'),('satisfaction','Satisfaction')]:
    z=plot_data[plot_data.outcome==y].copy()
    labs=(z.family+' / '+z.group+' (N='+z.n_pairs.astype(str)+')').tolist()
    yy=np.arange(len(z))
    fig,ax=plt.subplots(figsize=(11,max(5,.42*len(z)+1.5)))
    ax.errorbar(z.mean_diff,yy,xerr=[z.mean_diff-z.ci95_low,z.ci95_high-z.mean_diff],fmt='o',capsize=3)
    ax.axvline(0,linestyle='--',linewidth=1)
    ax.set_yticks(yy,labs)
    ax.set_xlabel('Mean paired difference: Avatar − Chat (95% t CI)')
    ax.set_title(f'Exploratory subgroup robustness — {name}')
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(OUT/f'13_forest_{y}.png',dpi=160,bbox_inches='tight')
    plt.show()

# %%
summary = {
 'sample':study_coverage,
 'primary':primary[['outcome','n_pairs','mean_diff','ci95_low','ci95_high','p_raw','p_holm_family']].to_dict('records'),
 'chat_preference':pref[pref.group=='chat_only'][['outcome','n_pairs','mean_diff','p_raw','p_holm_family']].to_dict('records'),
 'typed_only':typed[['outcome','n_pairs','mean_diff','p_raw','p_holm_family']].to_dict('records'),
 'emotional_absence':emot[['group','outcome','n_pairs','mean_diff','p_raw','p_holm_family']].to_dict('records'),
 'positive_joint_threshold_4':joint[joint.threshold==4.0].to_dict('records'),
 'knowledge_tost':eq[eq.outcome=='knowledge_accuracy'].to_dict('records'),
 'n_exploratory_wilcoxon_tests_in_notebook':len(all_new),
 'claims_not_supported':['more learning in Avatar','visual-only causal mechanism','no model-version effect','no fidelity issues'],
 'caveats':['post hoc','many searches','voice and emotion subgroups are post-treatment',
             'no knowledge pretest','LLM exposure categories require human validation']}
(OUT/'00_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print('Completed. Total output files:',len(list(OUT.iterdir())))
print('Main conclusion: stronger subjective Avatar experience; no demonstrated Avatar learning advantage.')
