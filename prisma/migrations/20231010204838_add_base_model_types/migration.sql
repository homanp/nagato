/*
  Warnings:

  - The values [LLAMA2] on the enum `BaseModelType` will be removed. If these variants are still used in the database, this will fail.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "BaseModelType_new" AS ENUM ('GPT_35_TURBO', 'LLAMA2_7B', 'LLAMA2_7B_CHAT', 'LLAMA2_13B', 'LLAMA2_13B_CHAT', 'LLAMA2_70B', 'LLAMA2_70B_CHAT');
ALTER TYPE "BaseModelType" RENAME TO "BaseModelType_old";
ALTER TYPE "BaseModelType_new" RENAME TO "BaseModelType";
DROP TYPE "BaseModelType_old";
COMMIT;
