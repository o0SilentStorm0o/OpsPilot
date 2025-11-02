import express, { Request, Response, NextFunction } from 'express';
import dotenv from 'dotenv';
import { logger } from './utils/logger';
import { metricsMiddleware, metricsEndpoint } from './monitoring/metrics';
import { analyzeLogs } from './workflows/analyzeLogs';
import { classifyIncident } from './workflows/classifyIncident';
import { recommendFix } from './workflows/recommendFix';
import { retrainModel } from './workflows/retrainModel';
import { mastraRouter } from './mastra/routes';
import workflowsRouter from './api/workflows';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(express.json());
app.use(metricsMiddleware);

// Health check
app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Metrics endpoint for Prometheus
app.get('/metrics', metricsEndpoint);

// Mastra AI routes
app.use('/api/mastra', mastraRouter);

// MastraAI Workflow routes (escalation from ML server)
app.use('/api/workflows', workflowsRouter);

// API Routes
app.post('/api/analyze-logs', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { logs } = req.body;
    
    if (!logs || !Array.isArray(logs)) {
      res.status(400).json({ error: 'Invalid logs format. Expected array of log entries.' });
      return;
    }

    logger.info(`Analyzing ${logs.length} log entries`);
    const result = await analyzeLogs(logs);
    
    res.json(result);
  } catch (error) {
    next(error);
  }
});

app.post('/api/classify-incident', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { incident } = req.body;
    
    if (!incident) {
      res.status(400).json({ error: 'Incident data required' });
      return;
    }

    logger.info('Classifying incident');
    const result = await classifyIncident(incident);
    
    res.json(result);
  } catch (error) {
    next(error);
  }
});

app.post('/api/recommend-fix', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const { incident, context } = req.body;
    
    if (!incident) {
      res.status(400).json({ error: 'Incident data required' });
      return;
    }

    logger.info('Generating fix recommendation');
    const result = await recommendFix(incident, context);
    
    res.json(result);
  } catch (error) {
    next(error);
  }
});

app.post('/api/retrain', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { feedbackData } = req.body;
    
    logger.info('Initiating model retraining');
    const result = await retrainModel(feedbackData);
    
    res.json(result);
  } catch (error) {
    next(error);
  }
});

// Error handling middleware
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`🚀 OpsPilot backend running on port ${PORT}`);
  logger.info(`📊 Metrics available at http://localhost:${PORT}/metrics`);
});

export default app;
