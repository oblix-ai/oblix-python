FROM python:3.13.3-alpine

# Install oblix package using pip
RUN pip install oblix

EXPOSE 8140

# Copy the entrypoint script into the container
COPY hook-models.json /oblix/hook-models.json
COPY entrypoint.sh /oblix/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /oblix/entrypoint.sh

# Run entrypoint.sh when the container launches
ENTRYPOINT ["/oblix/entrypoint.sh"]
