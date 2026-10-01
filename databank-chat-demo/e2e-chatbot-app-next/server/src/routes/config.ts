import {
  Router,
  type Request,
  type Response,
  type Router as RouterType,
} from 'express';
import { isDatabaseAvailable } from '@chat-template/db';

export const configRouter: RouterType = Router();

/**
 * GET /api/config - Get application configuration
 * Returns feature flags based on environment configuration
 */
configRouter.get('/', (_req: Request, res: Response) => {
  // MEMORY_MODE is set at deploy time (simple | shortterm | longterm) and drives
  // the deployment-mode badge in the header.
  const memoryMode = (process.env.MEMORY_MODE || 'simple').toLowerCase();
  res.json({
    memoryMode,
    features: {
      chatHistory: isDatabaseAvailable(),
      feedback: !!process.env.MLFLOW_EXPERIMENT_ID,
    },
  });
});
