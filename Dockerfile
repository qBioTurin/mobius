FROM tiagopeixoto/graph-tool:release-2.75 


# Aggiorna il sistema e installa pip, virtualenv e gcc (per fastcluster)
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm gcc python-pip python-virtualenv

# Imposta la directory di lavoro
WORKDIR /app

# Crea un ambiente virtuale che condivide le dipendenze col sistema sottostante (per graph-tool)
RUN python -m venv --system-site-packages  vdm 
#venv_de_mamt

# Attiva l'ambiente virtuale e installa le dipendenze
COPY requirements.txt .

### TODO: add openpyxl

RUN source vdm/bin/activate && pip install --no-cache-dir -r requirements.txt
# RUN source vdm/bin/activate && pip install --no-cache-dir \
#     'numpy<2' \
#     'scikit-learn<1.6' \
#     netrd \ 
#     networkx \
#     'statsmodels<0.15' \
#     pymoo \
#     dash_bio \
#     umap-learn \
#     markov_clustering \ 
#     scikit-network \
# # RUN source vdm/bin/activate && pip install --no-cache-dir \
#     xgboost \
#     imbalanced-learn \
#     pymongo \
#     shap 
# # RUN source vdm/bin/activate && pip install --no-cache-dir shap 

RUN source vdm/bin/activate && pip install --no-cache-dir kaleido==0.2.1


# Copia il resto del codice dell'applicazione nel container
WORKDIR /nicopad
COPY ./src /nicopad/

# Espone la porta utilizzata da Streamlit
EXPOSE 8501

# Comando per eseguire Streamlit all'interno dell'ambiente virtuale
### FORSE NON SERVE: --server.enableXsrfProtection false 
### --server.address 0.0.0.0 
CMD ["sh", "-c", "source /app/vdm/bin/activate && streamlit run --server.enableXsrfProtection false --server.address 0.0.0.0 --server.port=8501 mobius.py"]

