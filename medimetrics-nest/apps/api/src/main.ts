import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { ConfigService } from '@nestjs/config';
import * as cookieParser from 'cookie-parser';
import * as helmet from 'helmet';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/filters/http-exception.filter';
import { LoggingInterceptor } from './common/interceptors/logging.interceptor';
import { RedactInterceptor } from './common/interceptors/redact.interceptor';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  const configService = app.get(ConfigService);

  // Security
  app.use(helmet({
    contentSecurityPolicy: false, // Disabled for development
  }));
  app.use(cookieParser());

  // CORS
  app.enableCors({
    origin: configService.get('CORS_ORIGINS')?.split(',') || ['http://localhost:3000'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token'],
  });

  // Global pipes and filters
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      forbidNonWhitelisted: true,
      transformOptions: {
        enableImplicitConversion: true,
      },
    }),
  );

  app.useGlobalFilters(new HttpExceptionFilter());
  app.useGlobalInterceptors(
    new LoggingInterceptor(),
    new RedactInterceptor(),
  );

  // API Documentation
  const config = new DocumentBuilder()
    .setTitle('MediMetrics API')
    .setDescription('Medical Image Analysis Platform API')
    .setVersion('1.0.0')
    .addBearerAuth()
    .addCookieAuth('jwt')
    .addTag('auth', 'Authentication endpoints')
    .addTag('users', 'User management')
    .addTag('studies', 'Medical studies')
    .addTag('reports', 'Medical reports')
    .addTag('inference', 'AI inference')
    .addTag('dicom', 'DICOM operations')
    .addTag('admin', 'Administration')
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('docs', app, document);

  // Start server
  const port = configService.get('PORT') || 8000;
  await app.listen(port);

  console.log(`🚀 MediMetrics API is running on: http://localhost:${port}`);
  console.log(`📚 API Documentation: http://localhost:${port}/docs`);
  console.log(`📊 Metrics endpoint: http://localhost:${port}/metrics`);
}

bootstrap();