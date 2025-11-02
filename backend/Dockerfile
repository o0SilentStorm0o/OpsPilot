FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY backend/tsconfig.json ./backend/

# Install dependencies
RUN npm ci

# Copy source
COPY backend/src ./backend/src

# Build TypeScript
RUN npm run build

# Production stage
FROM node:20-alpine

# Install CA certificates and update certificate store (fixes TLS/blob errors)
RUN apk add --no-cache ca-certificates && update-ca-certificates

WORKDIR /app

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy built application
COPY --from=builder /app/backend/dist ./backend/dist

# Expose port
EXPOSE 3001

# Start application
CMD ["node", "backend/dist/main.js"]
