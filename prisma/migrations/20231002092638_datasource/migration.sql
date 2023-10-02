-- CreateEnum
CREATE TYPE "DatasourceType" AS ENUM ('TXT', 'PDF', 'CSV', 'MARKDOWN');

-- CreateEnum
CREATE TYPE "DatasourceStatus" AS ENUM ('IN_PROGRESS', 'DONE', 'FAILED');

-- CreateTable
CREATE TABLE "Datasource" (
    "id" TEXT NOT NULL,
    "content" TEXT,
    "status" "DatasourceStatus" NOT NULL DEFAULT 'IN_PROGRESS',
    "type" "DatasourceType" NOT NULL,
    "url" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Datasource_pkey" PRIMARY KEY ("id")
);
