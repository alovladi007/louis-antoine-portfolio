import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ThrottlerModule } from '@nestjs/throttler';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { StudiesModule } from './studies/studies.module';
import { ReportsModule } from './reports/reports.module';
import { InferenceModule } from './inference/inference.module';
import { DicomModule } from './dicom/dicom.module';
import { StorageModule } from './storage/storage.module';
import { AdminModule } from './admin/admin.module';
import { DatabaseModule } from './database/database.module';
import { MetricsController } from './app.metrics';
import { HealthController } from './health.controller';

@Module({
  imports: [
    // Configuration
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ['.env', '.env.local'],
    }),

    // Rate limiting
    ThrottlerModule.forRoot([{
      ttl: 60000,
      limit: 100,
    }]),

    // Database
    DatabaseModule,

    // Feature modules
    AuthModule,
    UsersModule,
    StudiesModule,
    ReportsModule,
    InferenceModule,
    DicomModule,
    StorageModule,
    AdminModule,
  ],
  controllers: [MetricsController, HealthController],
  providers: [],
})
export class AppModule {}