/*
  Warnings:

  - You are about to drop the column `finetuneId` on the `Datasource` table. All the data in the column will be lost.
  - You are about to drop the column `webhookUrl` on the `Datasource` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "Datasource" DROP COLUMN "finetuneId",
DROP COLUMN "webhookUrl",
ADD COLUMN     "finetune_id" TEXT,
ADD COLUMN     "webhook_url" TEXT;
