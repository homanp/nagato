-- CreateEnum
CREATE TYPE "ModelProviders" AS ENUM ('OPENAI', 'REPLICATE');

-- AlterTable
ALTER TABLE "Datasource" ADD COLUMN     "provider" TEXT NOT NULL DEFAULT 'OPENAI';
