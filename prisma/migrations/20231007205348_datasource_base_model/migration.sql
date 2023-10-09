-- CreateEnum
CREATE TYPE "BaseModelType" AS ENUM ('GPT_35_TURBO', 'LLAMA2');

-- AlterTable
ALTER TABLE "Datasource" ADD COLUMN     "base_model" TEXT NOT NULL DEFAULT 'GPT_35_TURBO';
