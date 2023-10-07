/*
  Warnings:

  - You are about to drop the column `finetune_id` on the `Datasource` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "Datasource" DROP COLUMN "finetune_id",
ADD COLUMN     "finetune" JSONB;
