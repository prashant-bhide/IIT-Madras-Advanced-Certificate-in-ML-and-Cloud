# Create the base image
FROM python:3.7-slim

# Change the working directory
WORKDIR /app/

# Install Dependency
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

# Copy local folder into the container
COPY app.py /app/
COPY df_KStable.csv /app/
COPY df_scn1_test.csv /app/
COPY scn1_gender_LR.pkl /app/
COPY scn1_age_LR.pkl /app/
###COPY templates/index.html /app/templates/index.html

# Set "python" as the entry point
ENTRYPOINT ["python"]

# Set the command as the script name
CMD ["app.py"]

#Expose the post 5000.
EXPOSE 5000
