import { Injectable, UnauthorizedException, BadRequestException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as bcrypt from 'bcrypt';
import * as speakeasy from 'speakeasy';
import { UsersService } from '../users/users.service';
import { User } from '../users/user.entity';
import { LoginDto } from './dto/login.dto';

@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
    private configService: ConfigService,
  ) {}

  async validateUser(email: string, password: string): Promise<User> {
    const user = await this.usersService.findByEmail(email);
    
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Check if account is locked
    if (user.lockedUntil && user.lockedUntil > new Date()) {
      throw new UnauthorizedException('Account is locked. Please try again later.');
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.passwordHash);
    
    if (!isPasswordValid) {
      // Increment failed login attempts
      await this.usersService.incrementFailedLoginAttempts(user.id);
      throw new UnauthorizedException('Invalid credentials');
    }

    // Reset failed login attempts on successful login
    await this.usersService.resetFailedLoginAttempts(user.id);
    
    return user;
  }

  async login(loginDto: LoginDto, ip: string) {
    const user = await this.validateUser(loginDto.email, loginDto.password);

    // Check if 2FA is enabled
    if (user.twofaEnabled) {
      if (!loginDto.totpCode) {
        return {
          requiresTwoFactor: true,
          message: 'Two-factor authentication required',
        };
      }

      const isValidTotp = this.verifyTotp(user.twofaSecretEnc, loginDto.totpCode);
      if (!isValidTotp) {
        throw new UnauthorizedException('Invalid two-factor code');
      }
    }

    // Update last login
    await this.usersService.updateLastLogin(user.id, ip);

    // Generate JWT
    const payload = {
      sub: user.id,
      email: user.email,
      role: user.role,
      orgId: user.orgId,
    };

    const accessToken = this.jwtService.sign(payload);

    return {
      accessToken,
      user: {
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        orgId: user.orgId,
      },
    };
  }

  async enable2FA(userId: string) {
    const secret = speakeasy.generateSecret({
      name: `MediMetrics (${userId})`,
      length: 32,
    });

    // In production, encrypt the secret before storing
    await this.usersService.update(userId, {
      twofaSecretEnc: secret.base32,
      twofaEnabled: false, // Will be enabled after verification
    });

    return {
      secret: secret.base32,
      qrCode: secret.otpauth_url,
    };
  }

  async verify2FA(userId: string, token: string) {
    const user = await this.usersService.findOne(userId);
    
    if (!user.twofaSecretEnc) {
      throw new BadRequestException('2FA not initialized');
    }

    const isValid = this.verifyTotp(user.twofaSecretEnc, token);
    
    if (!isValid) {
      throw new BadRequestException('Invalid verification code');
    }

    await this.usersService.update(userId, { twofaEnabled: true });
    
    return { message: '2FA enabled successfully' };
  }

  async disable2FA(userId: string, password: string) {
    const user = await this.usersService.findOne(userId);
    
    const isPasswordValid = await bcrypt.compare(password, user.passwordHash);
    if (!isPasswordValid) {
      throw new UnauthorizedException('Invalid password');
    }

    await this.usersService.update(userId, {
      twofaEnabled: false,
      twofaSecretEnc: null,
    });

    return { message: '2FA disabled successfully' };
  }

  private verifyTotp(secret: string, token: string): boolean {
    return speakeasy.totp.verify({
      secret,
      encoding: 'base32',
      token,
      window: 2,
    });
  }

  async refreshToken(userId: string) {
    const user = await this.usersService.findOne(userId);
    
    const payload = {
      sub: user.id,
      email: user.email,
      role: user.role,
      orgId: user.orgId,
    };

    return {
      accessToken: this.jwtService.sign(payload),
    };
  }
}