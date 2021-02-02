FROM cschranz/gpu-jupyter
COPY . .
EXPOSE 8501
RUN pip install -r requriements.txt
CMD streamlit run home-deploy.py