#include "rotator.h"
#include "encoder.h"

#include <time.h>

// 简单示例：生成随机正交矩阵，对向量做旋转并校验范数保持
static bool rotatorExampleUsage(const float *P, const float *vec, int D)
{
    float *vec_rotated = (float *)malloc(sizeof(float) * D);
    if (vec_rotated == NULL)
    {
        printf("分配旋转向量缓冲失败\n");
        return false;
    }

    rotateVector(P, vec, vec_rotated, D);

    float norm_original = 0.0f;
    float norm_rotated = 0.0f;
    for (int i = 0; i < D; i++)
    {
        norm_original += vec[i] * vec[i];
        norm_rotated += vec_rotated[i] * vec_rotated[i];
    }

    bool ok = fabsf(norm_original - norm_rotated) <= 1e-5f;
    if (!ok)
    {
        printf("旋转失败，L2范数不等: original=%f, rotated=%f\n", norm_original, norm_rotated);
    }

    free(vec_rotated);
    return ok;
}

int main(void)
{
    // 为随机正交矩阵设置随机种子
    srand((unsigned int)time(NULL));

    const size_t dim = 4;
    const float vec[4] = {0.1f, -0.3f, 0.5f, -0.2f};

    float *P = NULL;
    generateRotatorMatrix(&P, (int)dim);
    if (P == NULL)
    {
        printf("生成旋转矩阵失败\n");
        return 1;
    }

    if (rotatorExampleUsage(P, vec, (int)dim))
    {
        printf("旋转成功，L2 范数保持。\n");
    }

    // 展示旋转前后结合编码流程
    float *rotated = (float *)malloc(sizeof(float) * dim);
    if (rotated == NULL)
    {
        printf("分配旋转结果缓冲失败\n");
        destroyRotatorMatrix(&P);
        return 1;
    }

    rotateVector(P, vec, rotated, (int)dim);
    printf("对旋转后向量做 CAQ 编码示例:\n");
    encodeExample(rotated, dim, 9);

    free(rotated);
    destroyRotatorMatrix(&P);
    return 0;
}