"""Genera las 6 figuras de la red neuronal tal como salen en el informe (capítulos 5 y 6).

    python figuras_informe_red.py

Usa los modelos de modelos/, el historial de entrenamiento (modelos/historial.json) y las
celdas de resultados.ipynb que cargan los datos, la fórmula cerrada y las predicciones.
No entrena nada. Las figuras se guardan en figuras/informe/.
"""
import os, json, io, contextlib
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.font_manager as fm
fm.fontManager.ttflist=[f for f in fm.fontManager.ttflist if not (f.name=="Latin Modern Roman" and "lmroman10-" not in f.fname)]
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch, torch.nn as nn
from matplotlib.ticker import FuncFormatter
R=os.path.dirname(os.path.abspath(__file__))+"/"; os.chdir(R)
OUT=R+"figuras/informe/"; os.makedirs(OUT,exist_ok=True)
# ---- datos: celdas de resultados.ipynb (se buscan por su contenido, no por posición)
MARCAS=("datos = np.load","def bbva_vectorizada","prima_bbva = np.column_stack","class RedRecurrente","tabla_prima = pd.DataFrame")
nb=json.load(open("resultados.ipynb",encoding="utf-8")); G={}
with contextlib.redirect_stdout(io.StringIO()):
    for marca in MARCAS:
        celda=next(c for c in nb["cells"] if c["cell_type"]=="code" and marca in "".join(c["source"]))
        exec("".join(celda["source"]),G)
AZUL,ROJO,GRIS,OSC="#001391","#FF4B4B","#8A8D91","#4A4A55"
com=lambda v,d=4:f"{v:.{d}f}".replace(".",",").replace("-","−")
cf=lambda d:FuncFormatter(lambda v,_:com(v,d))
BASE={"font.family":"serif","font.serif":["Latin Modern Roman","DejaVu Serif"],"mathtext.fontset":"cm","font.size":10,
 "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"axes.axisbelow":True,"grid.color":"#E4E4EC","grid.linewidth":0.7,"axes.linewidth":0.8,"axes.edgecolor":"#444"}
W=6.3
hist=json.load(open(R+"modelos/historial.json"))
def save(fig,name): fig.savefig(OUT+name,dpi=250,facecolor="white"); plt.close(fig)

# 1 curvas base
with plt.rc_context(BASE):
    fig,ax=plt.subplots(figsize=(W*0.78,W*0.78*960/1890),layout="constrained")
    for k,c,l in [("base_rnn",GRIS,"RNN"),("base_lstm",ROJO,"LSTM"),("base_gru",AZUL,"GRU")]:
        v=hist[k]["val"]; ax.plot(range(1,len(v)+1),v,color=c,lw=1.5,label=l)
    ax.set(yscale="log",xlim=(1,40),xlabel="Época",ylabel="MSE en validación",xticks=[5,10,15,20,25,30,35,40]); ax.legend(frameon=False,loc="upper right")
    save(fig,"rnn_curvas_base.png")
# 2 curvas final
with plt.rc_context({**BASE,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":7.5,"axes.labelsize":9.5}):
    fig,axs=plt.subplots(1,2,figsize=(W*0.80,W*0.80*900/1980),sharey=True,layout="constrained")
    for ax,k,t in zip(axs,["r2_final_lstm","r2_final_gru"],["LSTM","GRU"]):
        h=hist[k]; e=range(1,len(h["val"])+1); b=h["val"].index(min(h["val"]))+1
        ax.plot(e,h["train"],color=AZUL,lw=1.2,label="Entrenamiento"); ax.plot(e,h["val"],color=ROJO,lw=1.2,label="Validación")
        ax.axvline(b,color=GRIS,ls="--",lw=1,label=f"Mejor época ({b})"); ax.set(yscale="log",xlabel="Época",title=t); ax.legend(frameon=False,loc="upper right")
        print(t,b,len(h["val"]))
    axs[0].set_ylabel("MSE"); save(fig,"rnn_curvas_final.png")
# datos de redes
CON=G["CONFIG"]; RR=G["RedRecurrente"]; idt=G["idx_test"]; logX=G["logX"]; payoff=G["payoff"]
x_te=torch.tensor(logX[idt],dtype=torch.float32); y_te=payoff[idt]
def cargar(t,s):
    r=RR(t,**CON[t]); r.load_state_dict(torch.load(R+f"modelos/r2_final_{t.lower()}"+(f"_s{s}" if s else "")+".pt")); return r.eval()
# 3 semillas MAE
mae={t:[np.abs(cargar(t,s)(x_te).squeeze(-1).detach().numpy()-y_te).mean() for s in (0,1,2)] for t in ("LSTM","GRU")}
print(mae)
with plt.rc_context({**BASE,"axes.grid":False}):
    fig,ax=plt.subplots(figsize=(W*0.62,W*0.62*960/1890),layout="constrained")
    ax.yaxis.grid(True)
    for xi,(t,c) in enumerate([("LSTM",ROJO),("GRU",AZUL)]):
        v=np.array(mae[t]); xs=xi+np.array([-0.12,0,0.12]); ax.scatter(xs,v,color=c,s=28,zorder=3)
        ax.hlines(v.mean(),xi-0.44,xi+0.44,color=c,lw=2,zorder=2); ax.text(xi+0.5,v.mean(),com(v.mean(),4),color=c,va="center",fontsize=9)
    ax.scatter([],[],color=GRIS,s=28,label="Semilla (0, 1 y 2)"); ax.plot([],[],color=GRIS,lw=2,label="Media de las tres")
    ax.set(xlim=(-0.6,1.85),ylim=(0,0.0025),ylabel="MAE en test",xticks=[0,1],xticklabels=["LSTM","GRU"]); ax.tick_params(axis="x",length=3)
    ax.yaxis.set_major_formatter(cf(4)); ax.legend(frameon=False,loc="upper right",fontsize=8)
    save(fig,"rnn_semillas_mae.png")
# 4 trimestres quitados
rows=[]
for t in ("LSTM","GRU"):
    for s in (0,1,2):
        r=cargar(t,s)
        with torch.no_grad():
            for L in (20,19,18,17,16):
                e=r(x_te[:,:L]).squeeze(-1).numpy()-y_te; rows.append({"red":t,"faltan":20-L,"RMSE":np.sqrt((e**2).mean())})
res3=pd.DataFrame(rows)
with plt.rc_context(BASE):
    fig,ax=plt.subplots(figsize=(W*0.78,W*0.78*1020/1890),layout="constrained")
    for t,c in [("LSTM",ROJO),("GRU",AZUL)]:
        m=res3[res3.red==t].groupby("faltan")["RMSE"].mean(); ax.plot(m.index,m.values,marker="o",ms=4.5,lw=1.5,color=c,label=t)
    ax.set(yscale="log",xlabel="Trimestres que faltan",ylabel="RMSE del payoff de T20",xticks=range(5)); ax.legend(frameon=False,loc="upper left")
    save(fig,"rnn_error_trimestres_quitados.png")
# 5 payoff k GRU
DESC=np.exp(-0.01*5); fil=[]
for t in ("LSTM","GRU"):
    for s in (0,1,2):
        r=cargar(t,s)
        with torch.no_grad():
            for j in range(1,21): fil.append({"red":t,"j":j,"prima":DESC*r(x_te[:,:j]).mean().item()})
prima_j=pd.DataFrame(fil).groupby(["j","red"])["prima"].mean().unstack(); pago=prima_j/DESC; pmc=y_te.mean()
KS=[16,17,18,19,20]; cols=plt.cm.viridis(np.linspace(0,0.9,5))
with plt.rc_context({**BASE,"legend.fontsize":7.5}):
    fig,ax=plt.subplots(figsize=(W*0.78,W*0.78*1110/1890),layout="constrained")
    for k,c in zip(KS,cols):
        p=pago.loc[k,"GRU"]; ax.plot([k,20],[p,p],color=c,marker="o",ms=3.5,lw=1.5,label=f"Datos hasta T{k}")
        ax.scatter(k,p,color=c,s=45,edgecolor="black",zorder=3)
        ax.annotate(f"{com(p)} (error {100*(p/pmc-1):+.2f} %)".replace(".",",").replace("-","−"),xy=(20,p),xytext=(20.3,p),va="center",fontsize=8,color=OSC)
    ax.scatter(20,pmc,color="black",marker="*",s=110,zorder=4,label=f"Montecarlo real ({com(pmc)})")
    ax.set(xlabel="La red ve datos hasta el trimestre",ylabel="Payoff de T20 predicho",xlim=(15.5,24.3),ylim=(0.277,0.3385)); ax.set_xticks(KS,[f"T{k}" for k in KS]); ax.yaxis.set_major_formatter(cf(3))
    ax.legend(frameon=False,loc="upper left"); save(fig,"rnn_payoff_k_gru.png")
# 6 prima t0 barras
tp=G["tabla_prima"]; p,d=tp["prima"].values,tp["diferencia %"].values; mc=p[1]
filas=[("Montecarlo\n100.000 sim.",p[0],GRIS,1.0,f"{100*(p[0]/mc-1):+.2f} %"),("Montecarlo\n10.000 de test",p[1],GRIS,0.5,"referencia"),
 ("LSTM\n(3 semillas)",p[2],ROJO,1.0,f"{d[2]:+.2f} %"),("GRU\n(3 semillas)",p[3],AZUL,1.0,f"{d[3]:+.2f} %"),("Fórmula\ncerrada",p[4],GRIS,0.25,f"unas {mc/p[4]:.1f} veces menos")]
with plt.rc_context({**BASE,"axes.grid":False,"axes.spines.left":False}):
    fig,ax=plt.subplots(figsize=(W*0.75,W*0.75*0.54))
    y=np.arange(5); ax.barh(y,0.40,height=0.62,color="#F1F1F6",zorder=1)
    for yi,(nom,v,c,a,txt) in zip(y,filas):
        ce="cerrada" in nom; ax.barh(yi,v,height=0.62,color=c,alpha=a,zorder=3,lw=0,hatch="///" if ce else None,edgecolor=GRIS if ce else c)
        ax.text(v+0.012,yi-0.17,com(v),va="center",fontweight="bold",fontsize=9.5,color=OSC if c==GRIS else c,zorder=5)
        ax.text(v+0.012,yi+0.16,txt.replace(".",","),va="center",fontsize=7.5,color=OSC,zorder=5)
    ax.axvline(mc,color=OSC,ls="--",lw=1,zorder=4); ax.text(mc,-0.62,"Montecarlo de test",ha="center",va="bottom",fontsize=8,color=OSC)
    ax.set_yticks(y,[f[0] for f in filas]); ax.invert_yaxis(); ax.set(xlim=(0,0.40),xlabel="Prima en t0"); ax.tick_params(axis="y",length=0)
    ax.xaxis.set_major_formatter(cf(2)); fig.subplots_adjust(left=0.22,right=0.965,bottom=0.17,top=0.92); save(fig,"rnn_prima_t0_final.png")
print("listo")
