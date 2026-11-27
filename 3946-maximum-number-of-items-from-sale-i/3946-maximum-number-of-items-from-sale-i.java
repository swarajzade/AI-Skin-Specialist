
class Solution {
    public int maximumSaleItems(int[][] items, int budget) {
        int n = items.length;

        int[] free = new int[n];

        // Count free items for each item type
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && items[j][0] % items[i][0] == 0) {
                    free[i]++;
                }
            }
        }

        // dp[cost] = maximum items using exactly cost
        int[] dp = new int[budget + 1];

        // -1 means this exact cost is not reachable
        java.util.Arrays.fill(dp, -1);
        dp[0] = 0;

        for (int i = 0; i < n; i++) {
            int price = items[i][1];
            int bonus = free[i];

            // Skip this item type
            int[] next = dp.clone();

            // States where we buy at least one copy of this type
            int[] activated = new int[budget + 1];
            java.util.Arrays.fill(activated, -1);

            // First copy: purchased item + free bonus
            for (int cost = price; cost <= budget; cost++) {
                if (dp[cost - price] != -1) {
                    activated[cost] =
                        dp[cost - price] + 1 + bonus;
                }
            }

            // Additional copies: only 1 item per copy
            for (int cost = price; cost <= budget; cost++) {
                if (activated[cost - price] != -1) {
                    activated[cost] = Math.max(
                        activated[cost],
                        activated[cost - price] + 1
                    );
                }
            }

            // Combine skip and buy choices
            for (int cost = 0; cost <= budget; cost++) {
                next[cost] = Math.max(
                    next[cost],
                    activated[cost]
                );
            }

            dp = next;
        }

        int answer = 0;

        for (int cost = 0; cost <= budget; cost++) {
            answer = Math.max(answer, dp[cost]);
        }

        return answer;
    }
}
