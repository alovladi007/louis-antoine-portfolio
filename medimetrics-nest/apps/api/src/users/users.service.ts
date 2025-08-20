import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import * as bcrypt from 'bcrypt';
import { User, UserRole } from './user.entity';

@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(User)
    private usersRepository: Repository<User>,
  ) {}

  async create(userData: Partial<User>): Promise<User> {
    const hashedPassword = await bcrypt.hash(userData.passwordHash || 'Demo123!', 10);
    const user = this.usersRepository.create({
      ...userData,
      passwordHash: hashedPassword,
    });
    return this.usersRepository.save(user);
  }

  async findAll(): Promise<User[]> {
    return this.usersRepository.find();
  }

  async findOne(id: string): Promise<User> {
    const user = await this.usersRepository.findOne({ where: { id } });
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    return user;
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.usersRepository.findOne({ where: { email } });
  }

  async update(id: string, updateData: Partial<User>): Promise<User> {
    await this.usersRepository.update(id, updateData);
    return this.findOne(id);
  }

  async updateLastLogin(id: string, ip: string): Promise<void> {
    await this.usersRepository.update(id, {
      lastLoginAt: new Date(),
      lastLoginIp: ip,
      failedLoginAttempts: 0,
    });
  }

  async incrementFailedLoginAttempts(id: string): Promise<void> {
    const user = await this.findOne(id);
    const attempts = user.failedLoginAttempts + 1;
    
    const updateData: Partial<User> = {
      failedLoginAttempts: attempts,
    };

    // Lock account after 5 failed attempts
    if (attempts >= 5) {
      const lockUntil = new Date();
      lockUntil.setMinutes(lockUntil.getMinutes() + 30);
      updateData.lockedUntil = lockUntil;
    }

    await this.usersRepository.update(id, updateData);
  }

  async resetFailedLoginAttempts(id: string): Promise<void> {
    await this.usersRepository.update(id, {
      failedLoginAttempts: 0,
      lockedUntil: null,
    });
  }

  async remove(id: string): Promise<void> {
    await this.usersRepository.delete(id);
  }
}