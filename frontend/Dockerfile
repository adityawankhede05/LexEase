FROM node:18-alpine

WORKDIR /app

# Copy package dependencies
COPY package*.json ./

# Install Node dependencies
RUN npm install

# Copy application files
COPY . .

# Expose port and start Vite dev server
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
